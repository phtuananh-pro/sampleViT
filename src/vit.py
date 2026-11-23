"""A deliberately small Vision Transformer for image classification.

The implementation follows the sequence in the accompanying chapter:

    image measurements
        -> non-overlapping image patches
        -> projected patch embeddings
        -> spatial/positional information
        -> Transformer blocks
        -> one image-level class prediction

The code is intentionally explicit rather than optimized. The comments use
the tensor notation ``(B, N, D)`` frequently used in the chapter:

* ``B`` = batch size, the number of images processed together.
* ``N`` = sequence length, here the number of visual tokens.
* ``D`` = embedding dimension, the length of one token vector.
* ``C`` = image channels, normally 3 for RGB.
* ``H`` and ``W`` = image height and width.

For the defaults in this file, a ``64 x 64`` RGB image becomes a sequence of
``(64 / 8) * (64 / 8) = 64`` patch tokens. Adding one learned class token
makes the Transformer sequence length ``65``.

Chapter VI.4 map: :class:`PatchEmbedding` covers §§2-4; learned spatial
positions are added in :class:`VisionTransformer` (§5); global token
interaction is in :class:`TransformerBlock` (§6); the class-token readout is
§8. The resulting patch count and full-attention cost follow §§9-11.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Turn a square image into a sequence of patch vectors.

    This class implements the chapter's first two visual objects:

    1. A *patch* is a small rectangular region of the original image.
    2. A *patch embedding* is a learned vector made from that region.

    We use a strided ``Conv2d`` as a compact implementation of two operations:

    ``extract non-overlapping patches + apply the same linear projection``

    A convolution with ``kernel_size=patch_size`` and
    ``stride=patch_size`` visits each patch exactly once. Its output channel
    dimension is the embedding dimension ``D``. This is the equivalence
    mentioned in the chapter: the input projection can be written as a
    convolution, even though the later Transformer computation is not a CNN.

    Chapter VI.4 §§2-4 distinguish a patch, its flattened measurements, and
    its learned embedding. This implementation uses one strided convolution
    to perform patch extraction and the shared learned projection together;
    it does not keep an intermediate raw flattened-patch tensor.
    """

    def __init__(self, image_size: int, patch_size: int, channels: int, embed_dim: int) -> None:
        """Set the fixed patch grid and shared patch-to-token projection.

        The number of grid positions is the ``N_p`` from VI.4 §3. A fixed
        square grid keeps this teaching implementation small; variable grids
        require a position strategy beyond the learned table used below (§13).
        """
        super().__init__()

        # The simple implementation assumes a regular grid with no partial
        # patches. For example, 64 / 8 gives an 8 x 8 patch grid. If the
        # division is not exact, we would need padding or a different image
        # size, so fail early with a useful message.
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        # N_patch = (H / P) * (W / P), where P is the patch size.
        # Because this example uses square images, H = W = image_size.
        self.num_patches = (image_size // patch_size) ** 2

        # Input:  (B, C, H, W)
        # Output: (B, D, H/P, W/P)
        #
        # Each output location corresponds to one image patch, and each
        # output channel is one coordinate of that patch's embedding vector.
        self.projection = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Project each non-overlapping image region into one patch embedding.

        This is the image-grid-to-visual-sequence transition from VI.4 §§2-4.
        The output is ``(B, N_p, D)``: patch embeddings, not yet
        position-aware or contextualized patch states.
        """

        # Before the projection, images are spatial grids:
        #     images.shape == (B, C, H, W)
        # Example: (8, 3, 64, 64)
        patches = self.projection(images)

        # After the convolution, the shape is (B, D, H/P, W/P).
        # Example: (8, 64, 8, 8).
        #
        # ``flatten(2)`` combines the two spatial grid dimensions into one:
        #     (B, D, H/P, W/P) -> (B, D, N_patch)
        #
        # ``transpose(1, 2)`` puts the sequence dimension before the feature
        # dimension, which is the format expected by batch_first=True
        # Transformer modules:
        #     (B, D, N_patch) -> (B, N_patch, D)
        #
        # No image-level meaning has been learned yet. These are projected
        # local measurements, one vector for each patch, in row-major order.
        return patches.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """One pre-normalized Transformer encoder block.

    A basic Transformer block has two sublayers:

    1. self-attention, allowing tokens to exchange information;
    2. a position-wise feed-forward network, transforming each token.

    Each sublayer has a residual connection. In this implementation the
    normalization is applied *before* each sublayer, so it is called a
    pre-norm block. The residual paths help information and gradients travel
    through several blocks without forcing every layer to relearn the full
    representation.

    This is the Part IV Transformer machinery reused by VI.4 §6. Full
    self-attention gives every visual state a direct route to every other
    state; §10 explains why interaction cost grows with sequence length.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Create one self-attention plus feed-forward Transformer block."""
        super().__init__()

        # LayerNorm operates on the final dimension D of each token and does
        # not mix information between different patches.
        self.norm1 = nn.LayerNorm(embed_dim)

        # PyTorch's MultiheadAttention performs the Query/Key/Value
        # projections and attention calculation internally. With
        # batch_first=True, the input and output shape is (B, N, D).
        #
        # In self-attention, the same token sequence is supplied as query,
        # key, and value. Therefore every patch can compare itself with all
        # other patches, including the class token once it is added.
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # The MLP is applied independently to every sequence position. It
        # mixes feature coordinates inside one token but does not directly
        # mix different patches; attention is the part that mixes positions.
        #
        # The temporary hidden width is usually larger than D. With the
        # default D=64 and mlp_ratio=4, it becomes 256.
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Let visual states exchange information while preserving their count.

        In VI.4 §6, this is where a patch embedding becomes a contextualized
        patch state. Attention mixes information across positions; the MLP
        transforms features within each position.
        """

        # tokens.shape is (B, N, D). Normalize before attention, but keep
        # the original tokens for the residual addition.
        normalized = self.norm1(tokens)

        # ``need_weights=False`` avoids returning the full attention matrix.
        # The matrix is useful for teaching/inspection, but omitting it here
        # keeps the training loop simpler and uses less memory.
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)

        # Residual connection around self-attention:
        #     tokens <- tokens + Attention(Norm(tokens))
        # A patch state can therefore retain its previous information while
        # also receiving information from other visual states.
        tokens = tokens + attended

        # Residual connection around the position-wise feed-forward network:
        #     tokens <- tokens + MLP(Norm(tokens))
        # The shape remains (B, N, D) throughout the block.
        return tokens + self.mlp(self.norm2(tokens))


class VisionTransformer(nn.Module):
    """A basic ViT with a learned class token and learned positions.

    The model deliberately follows the plain image-classification ViT path:

    ``image -> patch tokens -> class token and positions -> Transformer blocks``
    ``-> final class-token state -> linear classifier``

    The class token is not an image-specific object when it is initialized.
    It is one learned starting vector shared by every image. After attention
    has run, its *final activation* is image-specific because it has exchanged
    information with the patch tokens. That final state is used as the
    image-level representation for classification.

    The learned position table is also shared across examples. It tells the
    model which sequence slot corresponds to which location in the patch
    grid. Without position information, a Transformer would see a set of
    patch contents but would have much less information about their 2-D
    arrangement.

    Chapter VI.4 §5 is implemented by the learned position table; §8 is
    implemented by the prepended class token and final readout. This plain ViT
    uses one fixed input grid, leaving the sequence-length and variable-grid
    trade-offs from §§9-13 visible rather than hiding them in a larger system.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Assemble patch projection, spatial positions, encoder, and readout.

        The stored configuration lets inference rebuild exactly the same
        patch grid and positional-table shape as training.
        """
        super().__init__()

        # Save the architectural choices inside the model so a checkpoint can
        # rebuild exactly the same model during inference. These values are
        # configuration, not learned weights.
        self.config = {
            "image_size": image_size,
            "patch_size": patch_size,
            "channels": channels,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
        }

        # Patch embedding changes the image grid into N_patch visual tokens.
        self.patch_embedding = PatchEmbedding(image_size, patch_size, channels, embed_dim)

        # There is one extra token beyond the patches: the learned class token.
        # Therefore the positional table needs one position for [CLS] and one
        # position for every patch.
        num_tokens = self.patch_embedding.num_patches + 1

        # Shape (1, 1, D): one learned class-token vector. The first dimension
        # is 1 because the same parameter is expanded across each batch.
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Shape (1, 1 + N_patch, D): one learned position vector per sequence
        # slot. Position 0 belongs to the class token; the remaining positions
        # belong to image patches in the same order produced by the projection.
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.input_dropout = nn.Dropout(dropout)

        # Repeating blocks gives each patch more opportunities to interact
        # with distant patches and refine its representation. ``depth`` is
        # the number of repeated Transformer blocks, not image depth or patch
        # count.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # The final normalization is applied to the selected class-token
        # state before the classification head.
        self.norm = nn.LayerNorm(embed_dim)

        # The classifier maps one D-dimensional image representation to one
        # logit per class. Logits are unnormalized scores; CrossEntropyLoss in
        # train.py applies the appropriate softmax-like operation internally.
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Give the special learned tokens and head small starting values.

        The convolution, attention, MLP, and normalization layers already
        receive reasonable defaults from PyTorch. We explicitly initialize the
        parameters that are unique to this model so the important ViT objects
        are easy to identify.

        The class-token and position-table parameters correspond to VI.4 §§5
        and 8. Initialization itself is a training detail rather than a new
        visual representation stage.
        """

        # Small random values break symmetry: every position and the class
        # token must be allowed to learn a different role.
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

        # The classifier starts with small weights and zero class biases.
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized class score vector per image.

        Input:
            ``images`` with shape ``(B, C, H, W)``.

        Output:
            logits with shape ``(B, num_classes)``.

        The method makes the chapter's representation transitions visible:

        ``(B, C, H, W)``
            spatial image measurements;
        ``(B, N_patch, D)``
            projected patch states;
        ``(B, 1 + N_patch, D)``
            class token plus position-aware sequence;
        ``(B, 1 + N_patch, D)``
            contextualized states after Transformer blocks;
        ``(B, D)``
            final class-token image representation;
        ``(B, num_classes)``
            task-specific classification logits.

        The VI.4 path is explicit here: patch embeddings (§§2-4), position
        information (§5), contextualized states (§6), and a compressed
        image-level class-token readout (§8).
        """

        # Step 1: make one projected vector per image patch.
        tokens = self.patch_embedding(images)

        # Step 2: create one copy of the shared class-token parameter per
        # image. ``expand`` does not need to copy the underlying parameter; it
        # presents it as a batch of identical starting vectors.
        batch_size = tokens.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)

        # Step 3: prepend the class token. The sequence now has one special
        # readout state followed by all patch states:
        #     (B, N_patch, D) -> (B, 1 + N_patch, D)
        tokens = torch.cat((class_tokens, tokens), dim=1)

        # Step 4: add the learned position vector for every sequence slot.
        # Addition preserves the token shape. It combines content and spatial
        # identity before the first attention layer, matching h_i^0 = e_i + p_i
        # from the chapter.
        tokens = self.input_dropout(tokens + self.position_embedding)

        # Step 5: contextualize the whole visual sequence. With full
        # self-attention, every token can directly compare with every other
        # token in each block. This is why the attention score matrix has
        # roughly N^2 entries and why smaller patches become expensive.
        for block in self.blocks:
            tokens = block(tokens)

        # Step 6: select position 0, the final class-token state. It has had
        # the opportunity to gather information from all patches. The other
        # contextualized patch states still exist, but this classification task
        # compresses the image to one vector at the readout boundary.
        return self.classifier(self.norm(tokens[:, 0]))
