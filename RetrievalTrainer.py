import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from RetrievalDataset import BalancedPathologyBatchSampler
from RetrievalDataset import NIHRetrievalDataset
from RetrievalModels import RetrievalBackboneModel


def _normalize_patient_ids(patient_ids):

    normalized = []
    for value in patient_ids:
        if isinstance(value, bytes):
            normalized.append(value.decode("utf-8"))
        else:
            normalized.append(str(value))
    return normalized


def build_positive_mask(labels, patient_ids=None, positive_mode="label_overlap", treat_normal_as_positive=False):

    if positive_mode == "label_overlap":
        positiveMask = torch.matmul(labels, labels.t()) > 0
        if treat_normal_as_positive:
            normalMask = labels.sum(dim=1, keepdim=True) == 0
            positiveMask = positiveMask | torch.matmul(normalMask.float(), normalMask.float().t()).bool()
    elif positive_mode == "exact_label":
        positiveMask = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=-1)
    elif positive_mode == "same_patient":
        if patient_ids is None:
            raise ValueError("patient_ids are required when positive_mode='same_patient'")
        positiveMask = torch.tensor(
            [[anchor == candidate for candidate in patient_ids] for anchor in patient_ids],
            device=labels.device,
            dtype=torch.bool,
        )
    else:
        raise ValueError("Unsupported positive_mode: {0}".format(positive_mode))

    positiveMask.fill_diagonal_(False)
    return positiveMask


class MultiLabelSupConLoss(nn.Module):

    def __init__(self, temperature=0.07, positive_mode="label_overlap", treat_normal_as_positive=False):

        super(MultiLabelSupConLoss, self).__init__()
        self.temperature = temperature
        self.positive_mode = positive_mode
        self.treat_normal_as_positive = treat_normal_as_positive

    def forward(self, projections, labels, patient_ids=None):

        batchSize = projections.size(0)
        logits = torch.matmul(projections, projections.t()) / self.temperature
        logitsMask = torch.ones_like(logits) - torch.eye(batchSize, device=projections.device)
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        positiveMask = build_positive_mask(
            labels=labels,
            patient_ids=patient_ids,
            positive_mode=self.positive_mode,
            treat_normal_as_positive=self.treat_normal_as_positive,
        )

        positiveMask = positiveMask.float() * logitsMask
        expLogits = torch.exp(logits) * logitsMask
        logProb = logits - torch.log(expLogits.sum(dim=1, keepdim=True) + 1e-12)

        positiveCount = positiveMask.sum(dim=1)
        validRows = positiveCount > 0
        if validRows.sum() == 0:
            return projections.new_tensor(0.0)

        meanLogProbPos = (positiveMask * logProb).sum(dim=1)[validRows] / positiveCount[validRows]
        return -meanLogProbPos.mean()


class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):

        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)
        positiveProbabilities = probabilities
        negativeProbabilities = 1.0 - probabilities

        if self.clip is not None and self.clip > 0:
            negativeProbabilities = (negativeProbabilities + self.clip).clamp(max=1.0)

        positiveLoss = targets * torch.log(positiveProbabilities.clamp(min=self.eps))
        negativeLoss = (1.0 - targets) * torch.log(negativeProbabilities.clamp(min=self.eps))

        positiveWeights = torch.pow(1.0 - positiveProbabilities, self.gamma_pos) * targets
        negativeWeights = torch.pow(probabilities, self.gamma_neg) * (1.0 - targets)

        loss = positiveLoss * positiveWeights + negativeLoss * negativeWeights
        return -loss.mean()


class HardNegativeMemoryBank(object):

    def __init__(self, top_k=5, margin=0.1, positive_mode="label_overlap", treat_normal_as_positive=False):

        self.top_k = top_k
        self.margin = margin
        self.positive_mode = positive_mode
        self.treat_normal_as_positive = treat_normal_as_positive
        self.embeddings = None
        self.labels = None
        self.patient_ids = None
        self.indices = None

    def build(self, model, dataLoader, device):

        embeddings = []
        labels = []
        patientIds = []
        indices = []

        model.eval()
        with torch.no_grad():
            for batch in dataLoader:
                images = batch["image"].to(device, non_blocking=True)
                outputs = model(images)
                embeddings.append(outputs["embeddings"].detach().cpu())
                labels.append(batch["label"].cpu())
                patientIds.extend(_normalize_patient_ids(batch["patient_id"]))
                indices.extend(batch["index"].tolist())

        if embeddings:
            self.embeddings = torch.cat(embeddings, dim=0)
            self.labels = torch.cat(labels, dim=0)
            self.patient_ids = patientIds
            self.indices = indices

    def loss(self, batch_embeddings, batch_labels, batch_patient_ids, batch_indices):

        if self.embeddings is None or self.embeddings.size(0) == 0:
            return batch_embeddings.new_tensor(0.0)

        bankEmbeddings = self.embeddings.to(batch_embeddings.device)
        bankLabels = self.labels.to(batch_embeddings.device)
        similarity = torch.matmul(batch_embeddings, bankEmbeddings.t())
        sameSampleMask = torch.tensor(
            [[anchor == candidate for candidate in self.indices] for anchor in batch_indices],
            device=batch_embeddings.device,
            dtype=torch.bool,
        )

        if self.positive_mode == "same_patient":
            batchPatientIds = _normalize_patient_ids(batch_patient_ids)
            positiveMask = torch.tensor(
                [[anchor == candidate for candidate in self.patient_ids] for anchor in batchPatientIds],
                device=batch_embeddings.device,
                dtype=torch.bool,
            )
        else:
            bankPatientIds = None
            positiveMask = build_cross_positive_mask(
                anchor_labels=batch_labels,
                gallery_labels=bankLabels,
                anchor_patient_ids=batch_patient_ids,
                gallery_patient_ids=bankPatientIds if self.positive_mode != "same_patient" else self.patient_ids,
                positive_mode=self.positive_mode,
                treat_normal_as_positive=self.treat_normal_as_positive,
            )

        positiveMask = positiveMask & (~sameSampleMask)
        negativeMask = ~positiveMask
        negativeMask = negativeMask & (~sameSampleMask)
        if negativeMask.sum() == 0:
            return batch_embeddings.new_tensor(0.0)

        positiveSimilarity = similarity.masked_fill(~positiveMask, float("inf"))
        hardestPositive = positiveSimilarity.min(dim=1).values
        validPositive = torch.isfinite(hardestPositive)

        negativeSimilarity = similarity.masked_fill(~negativeMask, float("-inf"))
        hardNegativeValues, _ = negativeSimilarity.topk(k=min(self.top_k, negativeSimilarity.size(1)), dim=1)
        validNegative = torch.isfinite(hardNegativeValues).any(dim=1)

        valid = validPositive & validNegative
        if valid.sum() == 0:
            return batch_embeddings.new_tensor(0.0)

        hardestPositive = hardestPositive[valid].unsqueeze(1)
        hardNegativeValues = hardNegativeValues[valid]
        tripletLoss = torch.relu(hardNegativeValues - hardestPositive + self.margin)
        return tripletLoss.mean()


def build_cross_positive_mask(
    anchor_labels,
    gallery_labels,
    anchor_patient_ids=None,
    gallery_patient_ids=None,
    positive_mode="label_overlap",
    treat_normal_as_positive=False,
):

    if positive_mode == "label_overlap":
        positiveMask = torch.matmul(anchor_labels, gallery_labels.t()) > 0
        if treat_normal_as_positive:
            anchorNormalMask = anchor_labels.sum(dim=1, keepdim=True) == 0
            galleryNormalMask = gallery_labels.sum(dim=1, keepdim=True) == 0
            positiveMask = positiveMask | torch.matmul(anchorNormalMask.float(), galleryNormalMask.float().t()).bool()
        return positiveMask

    if positive_mode == "exact_label":
        return (anchor_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)).all(dim=-1)

    if positive_mode == "same_patient":
        if anchor_patient_ids is None or gallery_patient_ids is None:
            raise ValueError("patient ids are required for same_patient positives")

        anchorPatientIds = _normalize_patient_ids(anchor_patient_ids)
        galleryPatientIds = _normalize_patient_ids(gallery_patient_ids)
        return torch.tensor(
            [[anchor == candidate for candidate in galleryPatientIds] for anchor in anchorPatientIds],
            device=anchor_labels.device,
            dtype=torch.bool,
        )

    raise ValueError("Unsupported positive_mode: {0}".format(positive_mode))


def compute_map_at_k(relevance, k):

    if relevance.numel() == 0:
        return 0.0

    limited = relevance[:, :k].float()
    cumulative = torch.cumsum(limited, dim=1)
    ranks = torch.arange(1, limited.size(1) + 1, device=limited.device, dtype=limited.dtype).unsqueeze(0)
    precision = cumulative / ranks
    totalRelevant = relevance.float().sum(dim=1)
    denominator = torch.minimum(totalRelevant, torch.full_like(totalRelevant, float(k))).clamp(min=1.0)
    averagePrecision = (precision * limited).sum(dim=1) / denominator
    valid = totalRelevant > 0
    if valid.sum() == 0:
        return 0.0
    return averagePrecision[valid].mean().item()


def compute_ndcg_at_k(relevance, k):

    if relevance.numel() == 0:
        return 0.0

    limited = relevance[:, :k].float()
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=limited.device, dtype=limited.dtype))
    dcg = (limited * discount.unsqueeze(0)).sum(dim=1)

    idealLength = limited.size(1)
    positiveCounts = relevance.float().sum(dim=1).clamp(max=idealLength).long()
    idealRelevance = torch.zeros_like(limited)
    for rowIndex, count in enumerate(positiveCounts.tolist()):
        if count > 0:
            idealRelevance[rowIndex, :count] = 1.0
    idcg = (idealRelevance * discount.unsqueeze(0)).sum(dim=1)

    valid = idcg > 0
    if valid.sum() == 0:
        return 0.0

    ndcg = dcg[valid] / idcg[valid]
    return ndcg.mean().item()


class RetrievalTrainer(object):

    @staticmethod
    def _resolve_resize_size(imageSize, resizeSize=None):

        if resizeSize is not None:
            return resizeSize
        if imageSize == 384:
            return 432
        return 256

    @staticmethod
    def _build_train_transform(imageSize, resizeSize=None, randResize=False):

        resizeSize = RetrievalTrainer._resolve_resize_size(imageSize, resizeSize)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        cropTransform = (
            transforms.RandomCrop(imageSize, padding=4)
            if randResize
            else transforms.CenterCrop(imageSize)
        )

        return transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(resizeSize),
                cropTransform,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    @staticmethod
    def _create_model(device, architecture, use_pretrained, embeddingDim, projectionDim, classCount):

        model = RetrievalBackboneModel(
            architecture=architecture,
            use_pretrained=use_pretrained,
            embedding_dim=embeddingDim,
            projection_dim=projectionDim,
            class_count=classCount,
        ).to(device)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        return model

    @staticmethod
    def _build_eval_transform(imageSize, resizeSize=None):

        resizeSize = RetrievalTrainer._resolve_resize_size(imageSize, resizeSize)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(resizeSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                normalize,
            ]
        )

    @staticmethod
    def train(
        pathDirData,
        pathFileTrain,
        pathFileVal,
        architecture="resnet50",
        use_pretrained=True,
        classCount=14,
        batchSize=32,
        maxEpoch=20,
        imageSize=224,
        embeddingDim=128,
        projectionDim=128,
        temperature=0.07,
        retrievalWeight=1.0,
        classificationWeight=0.25,
        hardNegativeWeight=0.25,
        hardNegativeTopK=5,
        hardNegativeMargin=0.1,
        learningRate=1e-4,
        backboneLearningRate=None,
        headLearningRate=None,
        weightDecay=1e-5,
        samplesPerClass=4,
        grouping="pathology",
        positiveMode="label_overlap",
        treatNormalAsPositive=False,
        classificationLossName="asl",
        aslGammaNeg=4.0,
        aslGammaPos=1.0,
        aslClip=0.05,
        freezeBackboneEpochs=0,
        checkpointPath=None,
        outputDir="models",
        numWorkers=8,
        resizeSize=None,
        randResize=False,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = torch.cuda.is_available()

        model = RetrievalTrainer._create_model(
            device=device,
            architecture=architecture,
            use_pretrained=use_pretrained,
            embeddingDim=embeddingDim,
            projectionDim=projectionDim,
            classCount=classCount,
        )

        trainTransform = RetrievalTrainer._build_train_transform(
            imageSize=imageSize,
            resizeSize=resizeSize,
            randResize=randResize,
        )
        evalTransform = RetrievalTrainer._build_eval_transform(imageSize=imageSize, resizeSize=resizeSize)

        datasetTrain = NIHRetrievalDataset(pathDirData, pathFileTrain, transform=trainTransform)
        datasetTrainEval = NIHRetrievalDataset(pathDirData, pathFileTrain, transform=evalTransform)
        datasetVal = NIHRetrievalDataset(pathDirData, pathFileVal, transform=evalTransform)

        batchSamplerTrain = BalancedPathologyBatchSampler(
            datasetTrain,
            batch_size=batchSize,
            samples_per_group=samplesPerClass,
            grouping=grouping,
        )

        dataLoaderTrain = DataLoader(
            datasetTrain,
            batch_sampler=batchSamplerTrain,
            num_workers=numWorkers,
            pin_memory=torch.cuda.is_available(),
        )
        dataLoaderTrainEval = DataLoader(
            datasetTrainEval,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            pin_memory=torch.cuda.is_available(),
        )
        dataLoaderVal = DataLoader(
            datasetVal,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            pin_memory=torch.cuda.is_available(),
        )

        retrievalLoss = MultiLabelSupConLoss(
            temperature=temperature,
            positive_mode=positiveMode,
            treat_normal_as_positive=treatNormalAsPositive,
        )

        if classificationLossName == "asl":
            classificationLoss = AsymmetricLoss(
                gamma_neg=aslGammaNeg,
                gamma_pos=aslGammaPos,
                clip=aslClip,
            )
        elif classificationLossName == "bce":
            classificationLoss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unsupported classificationLossName: {0}".format(classificationLossName))

        hardNegativeMemory = HardNegativeMemoryBank(
            top_k=hardNegativeTopK,
            margin=hardNegativeMargin,
            positive_mode=positiveMode,
            treat_normal_as_positive=treatNormalAsPositive,
        )

        optimizer = RetrievalTrainer._build_optimizer(
            model=model,
            learningRate=learningRate,
            backboneLearningRate=backboneLearningRate,
            headLearningRate=headLearningRate,
            weightDecay=weightDecay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=maxEpoch)

        startEpoch = 0
        bestScore = -1.0

        if checkpointPath is not None:
            checkpoint = torch.load(checkpointPath, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            startEpoch = checkpoint.get("epoch", 0)
            bestScore = checkpoint.get("best_recall_at_1", -1.0)

        os.makedirs(outputDir, exist_ok=True)
        launchTimestamp = time.strftime("%d%m%Y-%H%M%S")

        for epochIndex in range(startEpoch, maxEpoch):
            RetrievalTrainer._set_backbone_trainable(model, epochIndex >= freezeBackboneEpochs)
            hardNegativeMemory.build(model, dataLoaderTrainEval, device)

            trainMetrics = RetrievalTrainer._train_epoch(
                model=model,
                dataLoader=dataLoaderTrain,
                optimizer=optimizer,
                retrievalLoss=retrievalLoss,
                classificationLoss=classificationLoss,
                hardNegativeMemory=hardNegativeMemory,
                retrievalWeight=retrievalWeight,
                classificationWeight=classificationWeight,
                hardNegativeWeight=hardNegativeWeight,
                device=device,
            )

            valMetrics = RetrievalTrainer.evaluate(
                model=model,
                dataLoader=dataLoaderVal,
                positiveMode=positiveMode,
                treatNormalAsPositive=treatNormalAsPositive,
                device=device,
            )

            scheduler.step()

            print(
                "Epoch [{0}/{1}] train_loss={2:.4f} retrieval_loss={3:.4f} cls_loss={4:.4f} hard_neg_loss={5:.4f} "
                "val_r1={6:.4f} val_r5={7:.4f} val_map10={8:.4f} val_ndcg10={9:.4f}".format(
                    epochIndex + 1,
                    maxEpoch,
                    trainMetrics["loss"],
                    trainMetrics["retrieval_loss"],
                    trainMetrics["classification_loss"],
                    trainMetrics["hard_negative_loss"],
                    valMetrics["recall_at_1"],
                    valMetrics["recall_at_5"],
                    valMetrics["map_at_10"],
                    valMetrics["ndcg_at_10"],
                )
            )

            if valMetrics["recall_at_1"] > bestScore:
                bestScore = valMetrics["recall_at_1"]
                checkpointOut = os.path.join(outputDir, "retrieval-{0}.pth.tar".format(launchTimestamp))
                torch.save(
                    {
                        "epoch": epochIndex + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_recall_at_1": bestScore,
                        "config": {
                            "architecture": architecture,
                            "embedding_dim": embeddingDim,
                            "projection_dim": projectionDim,
                            "image_size": imageSize,
                            "positive_mode": positiveMode,
                            "classification_loss": classificationLossName,
                        },
                    },
                    checkpointOut,
                )

    @staticmethod
    def test(
        pathDirData,
        pathFileTest,
        pathModel,
        architecture="resnet50",
        use_pretrained=False,
        classCount=14,
        batchSize=32,
        imageSize=224,
        embeddingDim=128,
        projectionDim=128,
        positiveMode="label_overlap",
        treatNormalAsPositive=False,
        numWorkers=8,
        resizeSize=None,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = torch.cuda.is_available()

        model = RetrievalTrainer._create_model(
            device=device,
            architecture=architecture,
            use_pretrained=use_pretrained,
            embeddingDim=embeddingDim,
            projectionDim=projectionDim,
            classCount=classCount,
        )

        checkpoint = torch.load(pathModel, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        evalTransform = RetrievalTrainer._build_eval_transform(imageSize=imageSize, resizeSize=resizeSize)
        datasetTest = NIHRetrievalDataset(pathDirData, pathFileTest, transform=evalTransform)
        dataLoaderTest = DataLoader(
            datasetTest,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            pin_memory=torch.cuda.is_available(),
        )

        metrics = RetrievalTrainer.evaluate(
            model=model,
            dataLoader=dataLoaderTest,
            device=device,
            positiveMode=positiveMode,
            treatNormalAsPositive=treatNormalAsPositive,
        )

        print(
            "test_r1={0:.4f} test_r5={1:.4f} test_r10={2:.4f} test_map10={3:.4f} test_ndcg10={4:.4f}".format(
                metrics["recall_at_1"],
                metrics["recall_at_5"],
                metrics["recall_at_10"],
                metrics["map_at_10"],
                metrics["ndcg_at_10"],
            )
        )

        return metrics

    @staticmethod
    def _build_optimizer(model, learningRate, backboneLearningRate, headLearningRate, weightDecay):

        backboneLearningRate = backboneLearningRate or learningRate
        headLearningRate = headLearningRate or learningRate

        modelModule = model.module if isinstance(model, torch.nn.DataParallel) else model
        parameterGroups = [
            {
                "params": modelModule.backbone.parameters(),
                "lr": backboneLearningRate,
            },
            {
                "params": modelModule.embeddingHead.parameters(),
                "lr": headLearningRate,
            },
            {
                "params": modelModule.projectionHead.parameters(),
                "lr": headLearningRate,
            },
            {
                "params": modelModule.classifierHead.parameters(),
                "lr": headLearningRate,
            },
        ]

        return optim.AdamW(parameterGroups, lr=learningRate, weight_decay=weightDecay)

    @staticmethod
    def _set_backbone_trainable(model, isTrainable):

        modelModule = model.module if isinstance(model, torch.nn.DataParallel) else model
        for parameter in modelModule.backbone.parameters():
            parameter.requires_grad = isTrainable

    @staticmethod
    def _train_epoch(
        model,
        dataLoader,
        optimizer,
        retrievalLoss,
        classificationLoss,
        hardNegativeMemory,
        retrievalWeight,
        classificationWeight,
        hardNegativeWeight,
        device,
    ):

        model.train()
        runningLoss = 0.0
        runningRetrievalLoss = 0.0
        runningClassificationLoss = 0.0
        runningHardNegativeLoss = 0.0
        batchCount = 0

        for batch in dataLoader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            patientIds = _normalize_patient_ids(batch["patient_id"])
            batchIndices = batch["index"].tolist()

            outputs = model(images)
            lossRetrieval = retrievalLoss(outputs["projections"], labels, patientIds)
            lossClassification = classificationLoss(outputs["logits"], labels)
            lossHardNegative = hardNegativeMemory.loss(outputs["embeddings"], labels, patientIds, batchIndices)
            loss = (
                retrievalWeight * lossRetrieval
                + classificationWeight * lossClassification
                + hardNegativeWeight * lossHardNegative
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            runningRetrievalLoss += lossRetrieval.item()
            runningClassificationLoss += lossClassification.item()
            runningHardNegativeLoss += lossHardNegative.item()
            batchCount += 1

        return {
            "loss": runningLoss / max(1, batchCount),
            "retrieval_loss": runningRetrievalLoss / max(1, batchCount),
            "classification_loss": runningClassificationLoss / max(1, batchCount),
            "hard_negative_loss": runningHardNegativeLoss / max(1, batchCount),
        }

    @staticmethod
    def evaluate(
        model,
        dataLoader,
        device,
        positiveMode="label_overlap",
        treatNormalAsPositive=False,
        ks=(1, 5, 10),
    ):

        model.eval()
        embeddings = []
        labels = []
        patientIds = []

        with torch.no_grad():
            for batch in dataLoader:
                images = batch["image"].to(device, non_blocking=True)
                outputs = model(images)

                embeddings.append(outputs["embeddings"])
                labels.append(batch["label"].to(device, non_blocking=True))
                patientIds.extend(_normalize_patient_ids(batch["patient_id"]))

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        similarity = torch.matmul(embeddings, embeddings.t())
        similarity.fill_diagonal_(-1e9)

        positiveMask = build_positive_mask(
            labels=labels,
            patient_ids=patientIds,
            positive_mode=positiveMode,
            treat_normal_as_positive=treatNormalAsPositive,
        )

        sortedIndices = similarity.argsort(dim=1, descending=True)
        rankedPositives = positiveMask.gather(1, sortedIndices)

        metrics = {}
        for k in ks:
            limitedK = min(k, rankedPositives.size(1))
            metrics["recall_at_{0}".format(k)] = rankedPositives[:, :limitedK].any(dim=1).float().mean().item()

        metrics["map_at_10"] = compute_map_at_k(rankedPositives, min(10, rankedPositives.size(1)))
        metrics["ndcg_at_10"] = compute_ndcg_at_k(rankedPositives, min(10, rankedPositives.size(1)))
        metrics["embedding_count"] = embeddings.size(0)
        metrics["positive_pair_fraction"] = positiveMask.float().mean().item()
        return metrics
