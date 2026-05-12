import importlib

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as tv_models


def _build_torchvision_model(builder_name, weight_enum_name, use_pretrained):

    builder = getattr(tv_models, builder_name, None)
    if builder is None:
        return None

    if use_pretrained:
        weightEnum = getattr(tv_models, weight_enum_name, None)
        if weightEnum is not None:
            return builder(weights=weightEnum.DEFAULT)
        return builder(pretrained=True)

    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)


def _build_timm_model(model_name, use_pretrained):

    try:
        timm = importlib.import_module("timm")
    except ImportError:
        return None

    return timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)


def _build_backbone_from_spec(spec, use_pretrained):

    source = spec["source"]
    if source == "torchvision":
        model = _build_torchvision_model(spec["builder"], spec["weights"], use_pretrained)
    elif source == "timm":
        model = _build_timm_model(spec["builder"], use_pretrained)
    else:
        raise ValueError("Unsupported backbone source: {0}".format(source))

    if model is None:
        return None, None

    if source == "torchvision":
        featureDim = spec["feature_dim"](model)
        spec["strip_head"](model)
        featureExtractor = spec.get("feature_extractor", model)
    else:
        featureDim = getattr(model, "num_features", None)
        if featureDim is None:
            raise ValueError("timm backbone missing num_features: {0}".format(spec["builder"]))
        featureExtractor = model

    return featureExtractor, featureDim


def _resnet_feature_dim(model):
    return model.fc.in_features


def _strip_resnet_head(model):
    model.fc = nn.Identity()


def _dense_feature_dim(model):
    return model.classifier.in_features


def _strip_dense_head(model):
    model.classifier = nn.Identity()


def _convnext_feature_dim(model):
    classifier = model.classifier
    if isinstance(classifier, nn.Sequential):
        for layer in reversed(classifier):
            if hasattr(layer, "in_features"):
                return layer.in_features
    raise ValueError("Unable to infer ConvNeXt feature dimension")


def _strip_convnext_head(model):
    model.classifier = nn.Identity()


def _vision_transformer_feature_dim(model):
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return model.head.in_features
    if hasattr(model, "heads"):
        for layer in reversed(model.heads):
            if hasattr(layer, "in_features"):
                return layer.in_features
    raise ValueError("Unable to infer ViT feature dimension")


def _strip_vit_head(model):
    if hasattr(model, "head"):
        model.head = nn.Identity()
    if hasattr(model, "heads"):
        model.heads = nn.Identity()


BACKBONE_SPECS = {
    "densenet121": {
        "source": "torchvision",
        "builder": "densenet121",
        "weights": "DenseNet121_Weights",
        "feature_dim": _dense_feature_dim,
        "strip_head": _strip_dense_head,
    },
    "densenet169": {
        "source": "torchvision",
        "builder": "densenet169",
        "weights": "DenseNet169_Weights",
        "feature_dim": _dense_feature_dim,
        "strip_head": _strip_dense_head,
    },
    "densenet201": {
        "source": "torchvision",
        "builder": "densenet201",
        "weights": "DenseNet201_Weights",
        "feature_dim": _dense_feature_dim,
        "strip_head": _strip_dense_head,
    },
    "resnet50": {
        "source": "torchvision",
        "builder": "resnet50",
        "weights": "ResNet50_Weights",
        "feature_dim": _resnet_feature_dim,
        "strip_head": _strip_resnet_head,
    },
    "convnextv2_tiny": {
        "source": "torchvision",
        "builder": "convnextv2_tiny",
        "weights": "ConvNeXt_V2_Tiny_Weights",
        "feature_dim": _convnext_feature_dim,
        "strip_head": _strip_convnext_head,
    },
    "convnextv2_base": {
        "source": "torchvision",
        "builder": "convnextv2_base",
        "weights": "ConvNeXt_V2_Base_Weights",
        "feature_dim": _convnext_feature_dim,
        "strip_head": _strip_convnext_head,
    },
    "dinov2_vits14": {
        "source": "torchvision",
        "builder": "dinov2_vits14",
        "weights": "DINOv2_ViTS14_Weights",
        "feature_dim": _vision_transformer_feature_dim,
        "strip_head": _strip_vit_head,
    },
    "dinov2_vitb14": {
        "source": "torchvision",
        "builder": "dinov2_vitb14",
        "weights": "DINOv2_ViTB14_Weights",
        "feature_dim": _vision_transformer_feature_dim,
        "strip_head": _strip_vit_head,
    },
    "dinov2_vitl14": {
        "source": "torchvision",
        "builder": "dinov2_vitl14",
        "weights": "DINOv2_ViTL14_Weights",
        "feature_dim": _vision_transformer_feature_dim,
        "strip_head": _strip_vit_head,
    },
    "convnextv2_tiny_timm": {
        "source": "timm",
        "builder": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    },
    "convnextv2_base_timm": {
        "source": "timm",
        "builder": "convnextv2_base.fcmae_ft_in22k_in1k",
    },
    "convnextv2_base_384_timm": {
        "source": "timm",
        "builder": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    },
    "dinov2_vits14_timm": {
        "source": "timm",
        "builder": "vit_small_patch14_dinov2.lvd142m",
    },
    "dinov2_vitb14_timm": {
        "source": "timm",
        "builder": "vit_base_patch14_dinov2.lvd142m",
    },
    "dinov2_vitl14_timm": {
        "source": "timm",
        "builder": "vit_large_patch14_dinov2.lvd142m",
    },
}


def list_supported_backbones():

    return sorted(BACKBONE_SPECS.keys())


def build_backbone(architecture, use_pretrained):

    spec = BACKBONE_SPECS.get(architecture)
    if spec is None:
        raise ValueError(
            "Unsupported retrieval architecture: {0}. Available: {1}".format(
                architecture, ", ".join(list_supported_backbones())
            )
        )

    featureExtractor, featureDim = _build_backbone_from_spec(spec, use_pretrained)
    if featureExtractor is None:
        raise RuntimeError(
            "Backbone '{0}' is configured but not available in this environment. "
            "Install a newer torchvision build or timm, or choose one of the locally available models.".format(
                architecture
            )
        )

    return featureExtractor, featureDim


class RetrievalBackboneModel(nn.Module):

    def __init__(
        self,
        architecture="resnet50",
        use_pretrained=True,
        embedding_dim=128,
        projection_dim=128,
        class_count=14,
        dropout=0.0,
    ):

        super(RetrievalBackboneModel, self).__init__()

        self.backbone, featureDim = build_backbone(architecture, use_pretrained)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.embeddingHead = nn.Linear(featureDim, embedding_dim)
        self.projectionHead = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifierHead = nn.Linear(embedding_dim, class_count)

    def forward(self, x):

        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.dim() > 2:
            features = torch.flatten(features, 1)

        embeddings = self.embeddingHead(self.dropout(features))
        normalizedEmbeddings = func.normalize(embeddings, dim=1)
        projections = self.projectionHead(normalizedEmbeddings)
        normalizedProjections = func.normalize(projections, dim=1)
        logits = self.classifierHead(embeddings)

        return {
            "features": features,
            "embeddings": normalizedEmbeddings,
            "projections": normalizedProjections,
            "logits": logits,
        }
