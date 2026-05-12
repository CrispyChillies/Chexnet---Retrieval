import argparse
import time

def add_shared_dataset_args(parser):

    parser.add_argument("--data-dir", default="./database")
    parser.add_argument("--class-count", type=int, default=14)


def add_shared_retrieval_args(parser):

    add_shared_dataset_args(parser)
    parser.add_argument("--architecture", default="resnet50")
    parser.add_argument("--use-pretrained", dest="use_pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false")
    parser.set_defaults(use_pretrained=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--positive-mode", choices=["label_overlap", "exact_label", "same_patient"], default="label_overlap")
    parser.add_argument("--treat-normal-as-positive", action="store_true")
    parser.add_argument("--rand-resize", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)


def run_classification_train(args):

    from ChexnetTrainer import ChexnetTrainer

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + "-" + timestampTime

    print("Training NN architecture =", args.architecture)
    ChexnetTrainer.train(
        args.data_dir,
        args.train_file,
        args.val_file,
        args.architecture,
        args.use_pretrained,
        args.class_count,
        args.batch_size,
        args.max_epoch,
        args.resize,
        args.crop,
        timestampLaunch,
        args.checkpoint,
    )


def run_classification_test(args):

    from ChexnetTrainer import ChexnetTrainer

    ChexnetTrainer.test(
        args.data_dir,
        args.test_file,
        args.model_path,
        args.architecture,
        args.class_count,
        args.use_pretrained,
        args.batch_size,
        args.resize,
        args.crop,
        "",
    )


def run_retrieval_split(args):

    from SplitBuilder import build_patient_disjoint_splits

    outputPaths = build_patient_disjoint_splits(
        input_files=args.input_files,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        prefix=args.prefix,
    )
    print(outputPaths)


def run_retrieval_train(args):

    from RetrievalTrainer import RetrievalTrainer

    RetrievalTrainer.train(
        pathDirData=args.data_dir,
        pathFileTrain=args.train_file,
        pathFileVal=args.val_file,
        architecture=args.architecture,
        use_pretrained=args.use_pretrained,
        classCount=args.class_count,
        batchSize=args.batch_size,
        maxEpoch=args.max_epoch,
        imageSize=args.image_size,
        resizeSize=args.resize_size,
        embeddingDim=args.embedding_dim,
        projectionDim=args.projection_dim,
        temperature=args.temperature,
        retrievalWeight=args.retrieval_weight,
        classificationWeight=args.classification_weight,
        hardNegativeWeight=args.hard_negative_weight,
        hardNegativeTopK=args.hard_negative_topk,
        hardNegativeMargin=args.hard_negative_margin,
        learningRate=args.learning_rate,
        backboneLearningRate=args.backbone_learning_rate,
        headLearningRate=args.head_learning_rate,
        weightDecay=args.weight_decay,
        samplesPerClass=args.samples_per_class,
        grouping=args.grouping,
        positiveMode=args.positive_mode,
        treatNormalAsPositive=args.treat_normal_as_positive,
        classificationLossName=args.classification_loss,
        aslGammaNeg=args.asl_gamma_neg,
        aslGammaPos=args.asl_gamma_pos,
        aslClip=args.asl_clip,
        freezeBackboneEpochs=args.freeze_backbone_epochs,
        checkpointPath=args.checkpoint,
        outputDir=args.output_dir,
        numWorkers=args.num_workers,
        randResize=args.rand_resize,
    )


def run_retrieval_test(args):

    from RetrievalTrainer import RetrievalTrainer

    RetrievalTrainer.test(
        pathDirData=args.data_dir,
        pathFileTest=args.test_file,
        pathModel=args.model_path,
        architecture=args.architecture,
        use_pretrained=args.use_pretrained,
        classCount=args.class_count,
        batchSize=args.batch_size,
        imageSize=args.image_size,
        resizeSize=args.resize_size,
        embeddingDim=args.embedding_dim,
        projectionDim=args.projection_dim,
        positiveMode=args.positive_mode,
        treatNormalAsPositive=args.treat_normal_as_positive,
        numWorkers=args.num_workers,
    )


def build_parser():

    parser = argparse.ArgumentParser(description="CheXNet classification and NIH retrieval training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classificationTrainParser = subparsers.add_parser("classification-train")
    add_shared_dataset_args(classificationTrainParser)
    classificationTrainParser.add_argument("--train-file", default="./dataset/train_1.txt")
    classificationTrainParser.add_argument("--val-file", default="./dataset/val_1.txt")
    classificationTrainParser.add_argument(
        "--architecture",
        choices=["DENSE-NET-121", "DENSE-NET-169", "DENSE-NET-201"],
        default="DENSE-NET-121",
    )
    classificationTrainParser.add_argument("--use-pretrained", dest="use_pretrained", action="store_true")
    classificationTrainParser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false")
    classificationTrainParser.set_defaults(use_pretrained=True)
    classificationTrainParser.add_argument("--batch-size", type=int, default=16)
    classificationTrainParser.add_argument("--max-epoch", type=int, default=100)
    classificationTrainParser.add_argument("--resize", type=int, default=256)
    classificationTrainParser.add_argument("--crop", type=int, default=224)
    classificationTrainParser.add_argument("--checkpoint", default=None)
    classificationTrainParser.set_defaults(func=run_classification_train)

    classificationTestParser = subparsers.add_parser("classification-test")
    add_shared_dataset_args(classificationTestParser)
    classificationTestParser.add_argument("--test-file", default="./dataset/test_1.txt")
    classificationTestParser.add_argument("--model-path", default="./models/m-25012018-123527.pth.tar")
    classificationTestParser.add_argument(
        "--architecture",
        choices=["DENSE-NET-121", "DENSE-NET-169", "DENSE-NET-201"],
        default="DENSE-NET-121",
    )
    classificationTestParser.add_argument("--use-pretrained", dest="use_pretrained", action="store_true")
    classificationTestParser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false")
    classificationTestParser.set_defaults(use_pretrained=True)
    classificationTestParser.add_argument("--batch-size", type=int, default=16)
    classificationTestParser.add_argument("--resize", type=int, default=256)
    classificationTestParser.add_argument("--crop", type=int, default=224)
    classificationTestParser.set_defaults(func=run_classification_test)

    retrievalSplitParser = subparsers.add_parser("retrieval-split")
    retrievalSplitParser.add_argument(
        "--input-files",
        nargs="+",
        default=["./dataset/train_1.txt", "./dataset/val_1.txt", "./dataset/test_1.txt"],
    )
    retrievalSplitParser.add_argument("--output-dir", default="./dataset")
    retrievalSplitParser.add_argument("--train-ratio", type=float, default=0.7)
    retrievalSplitParser.add_argument("--val-ratio", type=float, default=0.1)
    retrievalSplitParser.add_argument("--seed", type=int, default=0)
    retrievalSplitParser.add_argument("--prefix", default="retrieval_patient")
    retrievalSplitParser.set_defaults(func=run_retrieval_split)

    retrievalTrainParser = subparsers.add_parser("retrieval-train")
    add_shared_retrieval_args(retrievalTrainParser)
    retrievalTrainParser.add_argument("--train-file", default="./dataset/retrieval_patient_train.txt")
    retrievalTrainParser.add_argument("--val-file", default="./dataset/retrieval_patient_val.txt")
    retrievalTrainParser.add_argument("--max-epoch", type=int, default=20)
    retrievalTrainParser.add_argument("--temperature", type=float, default=0.07)
    retrievalTrainParser.add_argument("--retrieval-weight", type=float, default=1.0)
    retrievalTrainParser.add_argument("--classification-weight", type=float, default=0.25)
    retrievalTrainParser.add_argument("--hard-negative-weight", type=float, default=0.25)
    retrievalTrainParser.add_argument("--hard-negative-topk", type=int, default=5)
    retrievalTrainParser.add_argument("--hard-negative-margin", type=float, default=0.1)
    retrievalTrainParser.add_argument("--learning-rate", type=float, default=1e-4)
    retrievalTrainParser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    retrievalTrainParser.add_argument("--head-learning-rate", type=float, default=1e-4)
    retrievalTrainParser.add_argument("--weight-decay", type=float, default=1e-5)
    retrievalTrainParser.add_argument("--samples-per-class", type=int, default=4)
    retrievalTrainParser.add_argument("--grouping", choices=["pathology", "exact_label", "patient"], default="pathology")
    retrievalTrainParser.add_argument("--classification-loss", choices=["asl", "bce"], default="asl")
    retrievalTrainParser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    retrievalTrainParser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    retrievalTrainParser.add_argument("--asl-clip", type=float, default=0.05)
    retrievalTrainParser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    retrievalTrainParser.add_argument("--checkpoint", default=None)
    retrievalTrainParser.add_argument("--output-dir", default="./models")
    retrievalTrainParser.set_defaults(func=run_retrieval_train)

    retrievalTestParser = subparsers.add_parser("retrieval-test")
    add_shared_retrieval_args(retrievalTestParser)
    retrievalTestParser.add_argument("--test-file", default="./dataset/retrieval_patient_test.txt")
    retrievalTestParser.add_argument("--model-path", required=True)
    retrievalTestParser.set_defaults(func=run_retrieval_test)

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
