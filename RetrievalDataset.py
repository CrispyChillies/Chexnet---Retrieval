import os
import random
from collections import defaultdict
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler


def extract_patient_id(relative_path):

    filename = os.path.basename(relative_path)
    return filename.split("_")[0]


def label_signature(label_tensor):

    return tuple(int(value) for value in label_tensor.tolist())


class NIHRetrievalDataset(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, transform=None):

        self.pathImageDirectory = pathImageDirectory
        self.pathDatasetFile = pathDatasetFile
        self.transform = transform
        self.records = []
        self.class_to_indices = defaultdict(list)
        self.patient_to_indices = defaultdict(list)
        self.signature_to_indices = defaultdict(list)
        self.normal_indices = []
        self.label_count = None

        with open(pathDatasetFile, "r") as fileDescriptor:
            for line in fileDescriptor:
                line = line.strip()
                if not line:
                    continue

                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = torch.tensor([int(item) for item in lineItems[1:]], dtype=torch.float32)
                patientId = extract_patient_id(lineItems[0])

                if self.label_count is None:
                    self.label_count = len(imageLabel)

                index = len(self.records)
                positiveClasses = []
                for classIndex, labelValue in enumerate(imageLabel.tolist()):
                    if labelValue > 0:
                        self.class_to_indices[classIndex].append(index)
                        positiveClasses.append(classIndex)

                if not positiveClasses:
                    self.normal_indices.append(index)

                signature = label_signature(imageLabel)
                self.patient_to_indices[patientId].append(index)
                self.signature_to_indices[signature].append(index)
                self.records.append(
                    {
                        "image_path": imagePath,
                        "relative_path": lineItems[0],
                        "label": imageLabel,
                        "patient_id": patientId,
                        "signature": signature,
                        "positive_classes": positiveClasses,
                    }
                )

        if self.label_count is None:
            raise ValueError("Dataset file is empty: {0}".format(pathDatasetFile))

    def __getitem__(self, index):

        record = self.records[index]
        imageData = Image.open(record["image_path"]).convert("RGB")

        if self.transform is not None:
            imageData = self.transform(imageData)

        return {
            "image": imageData,
            "label": record["label"],
            "index": index,
            "patient_id": record["patient_id"],
            "path": record["relative_path"],
        }

    def __len__(self):

        return len(self.records)

    def build_groups(self, grouping="pathology", include_normal=True):

        groups = []

        if grouping == "pathology":
            for classIndex, indices in sorted(self.class_to_indices.items()):
                if indices:
                    groups.append(("class_{0}".format(classIndex), indices))
            if include_normal and self.normal_indices:
                groups.append(("normal", self.normal_indices))
        elif grouping == "patient":
            for patientId, indices in sorted(self.patient_to_indices.items()):
                if len(indices) > 1:
                    groups.append((patientId, indices))
        elif grouping == "exact_label":
            for signature, indices in sorted(self.signature_to_indices.items()):
                if sum(signature) > 0 and len(indices) > 1:
                    groups.append((signature, indices))
            if include_normal and self.normal_indices:
                groups.append(("normal", self.normal_indices))
        else:
            raise ValueError("Unsupported grouping mode: {0}".format(grouping))

        return groups


class BalancedPathologyBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_group=4,
        batches_per_epoch=None,
        include_normal=True,
        grouping="pathology",
    ):

        if batch_size % samples_per_group != 0:
            raise ValueError("batch_size must be divisible by samples_per_group")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_group = samples_per_group
        self.groups_per_batch = batch_size // samples_per_group
        self.include_normal = include_normal
        self.grouping = grouping
        self.batches_per_epoch = batches_per_epoch or max(1, len(dataset) // batch_size)

        self.available_groups = dataset.build_groups(grouping=grouping, include_normal=include_normal)
        if not self.available_groups:
            raise ValueError("No groups available for retrieval sampling with grouping={0}".format(grouping))

    def __iter__(self):

        for _ in range(self.batches_per_epoch):
            batchIndices = []
            usedIndices = set()

            if len(self.available_groups) >= self.groups_per_batch:
                chosenGroups = random.sample(self.available_groups, self.groups_per_batch)
            else:
                chosenGroups = random.choices(self.available_groups, k=self.groups_per_batch)

            for _, candidateIndices in chosenGroups:
                chosen = self._sample_unique_indices(candidateIndices, self.samples_per_group, usedIndices)
                batchIndices.extend(chosen)
                usedIndices.update(chosen)

            if len(batchIndices) < self.batch_size:
                remaining = [index for index in range(len(self.dataset)) if index not in usedIndices]
                if len(remaining) >= self.batch_size - len(batchIndices):
                    batchIndices.extend(random.sample(remaining, self.batch_size - len(batchIndices)))
                else:
                    batchIndices.extend(remaining)
                    while len(batchIndices) < self.batch_size:
                        batchIndices.append(random.randrange(len(self.dataset)))

            yield batchIndices[: self.batch_size]

    def __len__(self):

        return self.batches_per_epoch

    @staticmethod
    def _sample_unique_indices(candidateIndices, count, usedIndices):

        available = [index for index in candidateIndices if index not in usedIndices]
        if len(available) >= count:
            return random.sample(available, count)

        result = list(available)
        if not candidateIndices:
            return result

        while len(result) < count:
            result.append(random.choice(candidateIndices))

        return result
