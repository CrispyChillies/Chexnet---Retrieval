import os
import random
from collections import defaultdict

from RetrievalDataset import extract_patient_id


def _read_entries(input_files):

    entries = []
    for inputFile in input_files:
        with open(inputFile, "r") as fileDescriptor:
            for line in fileDescriptor:
                line = line.strip()
                if not line:
                    continue

                lineItems = line.split()
                relativePath = lineItems[0]
                labelValues = lineItems[1:]
                entries.append(
                    {
                        "line": line,
                        "relative_path": relativePath,
                        "patient_id": extract_patient_id(relativePath),
                        "labels": labelValues,
                    }
                )
    return entries


def build_patient_disjoint_splits(
    input_files,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.1,
    seed=0,
    prefix="retrieval_patient",
):

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1")

    entries = _read_entries(input_files)
    patientToEntries = defaultdict(list)
    for entry in entries:
        patientToEntries[entry["patient_id"]].append(entry["line"])

    patientIds = list(patientToEntries.keys())
    random.Random(seed).shuffle(patientIds)

    patientCount = len(patientIds)
    trainCutoff = int(patientCount * train_ratio)
    valCutoff = trainCutoff + int(patientCount * val_ratio)

    splitPatients = {
        "train": set(patientIds[:trainCutoff]),
        "val": set(patientIds[trainCutoff:valCutoff]),
        "test": set(patientIds[valCutoff:]),
    }

    os.makedirs(output_dir, exist_ok=True)
    outputPaths = {
        splitName: os.path.join(output_dir, "{0}_{1}.txt".format(prefix, splitName))
        for splitName in splitPatients
    }

    for splitName, outputPath in outputPaths.items():
        with open(outputPath, "w") as fileDescriptor:
            for patientId in sorted(splitPatients[splitName]):
                for line in patientToEntries[patientId]:
                    fileDescriptor.write(line + "\n")

    return outputPaths
