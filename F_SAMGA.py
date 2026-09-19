"""Frequency-guided Structure-Aware Multi-Granularity Alignment (F-SAMGA).

This file rewrites the original SAMGA alignment module into the design used with
FG-SGE. The module no longer accepts raw image feature maps as the main visual
input. It expects graph-enhanced visual evidence, node-level DWT frequency
responses, multi-relation graph edges, and text tokens.

Expected inputs
---------------
visual_evidence: Tensor[B, N, D]
    Graph-enhanced visual evidence tokens from FG-SGE.

freq_desc: Tensor[B, N, 4]
    Node-level DWT sub-band response descriptor in the order [LL, LH, HL, HH].

relation_graphs: Dict[str, Tuple[LongTensor, FloatTensor]]
    Sparse dynamic multi-relation graph. Each key in {"adj", "sem", "co", "freq"}
    maps to a tuple (index, weight):
        index  : LongTensor[B, N, K]
        weight : FloatTensor[B, N, K]
    where index[b, i] stores the Top-K neighbor indices of node i for that
    relation, and weight[b, i] stores the corresponding edge weights.

text_tokens: Tensor[B, M, D]
    Token-level text representations from a text encoder.

Output
------
By default returns a scalar training loss. If return_details=True, returns
(loss, output_dict), where output_dict contains the scene/entity/composition
alignment scores, transport plans, and attention maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropicOT(nn.Module):
    """Log-domain Sinkhorn solver for a given cost matrix.

    Args:
        eps: Entropy regularization strength.
        max_iter: Number of Sinkhorn iterations.
        tol: Optional early stopping tolerance.
    """

    def __init__(self, eps=0.07, max_iter=50, tol=1e-4):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, cost):
        """Compute entropic transport distance and plan.

        Args:
            cost: Tensor[B, N, M]. Lower values indicate better matching.

        Returns:
            distance: Tensor[B].
            plan: Tensor[B, N, M].
        """
        if cost.dim() != 3:
            raise ValueError("cost must have shape [B, N, M].")

        B, N, M = cost.shape
        dtype = cost.dtype
        device = cost.device

        log_a = torch.full((B, N), -torch.log(torch.tensor(float(N), device=device, dtype=dtype)), device=device, dtype=dtype)
        log_b = torch.full((B, M), -torch.log(torch.tensor(float(M), device=device, dtype=dtype)), device=device, dtype=dtype)

        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)

        eps = max(float(self.eps), 1e-6)
        for _ in range(self.max_iter):
            u_prev = u
            u = eps * (log_a - torch.logsumexp((v.unsqueeze(1) - cost) / eps, dim=2))
            v = eps * (log_b - torch.logsumexp((u.unsqueeze(2) - cost) / eps, dim=1))
            if torch.max(torch.abs(u - u_prev)).item() < self.tol:
                break

        plan = torch.exp((u.unsqueeze(2) + v.unsqueeze(1) - cost) / eps)
        distance = torch.sum(plan * cost, dim=(1, 2))
        return distance, plan


class TextGranularityExtractor(nn.Module):
    """Extract scene/entity/composition textual semantics with learnable queries.

    This avoids external POS tags or phrase parsers. The scene query extracts a
    sentence-level semantic token, entity queries extract noun/object-related
    semantics, and composition queries extract phrase/relation-level semantics.
    """

    def __init__(self, embed_dim, num_heads=8, num_entity_queries=16, num_comp_queries=16, dropout=0.1):
        super().__init__()
        self.scene_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.entity_queries = nn.Parameter(torch.randn(1, num_entity_queries, embed_dim) * 0.02)
        self.comp_queries = nn.Parameter(torch.randn(1, num_comp_queries, embed_dim) * 0.02)

        self.scene_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.entity_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.comp_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_scene = nn.LayerNorm(embed_dim)
        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_comp = nn.LayerNorm(embed_dim)

    def forward(self, text_tokens, text_padding_mask=None):
        B = text_tokens.shape[0]
        scene_q = self.scene_query.expand(B, -1, -1)
        entity_q = self.entity_queries.expand(B, -1, -1)
        comp_q = self.comp_queries.expand(B, -1, -1)

        scene_tokens, scene_attn = self.scene_attn(scene_q, text_tokens, text_tokens, key_padding_mask=text_padding_mask)
        entity_tokens, entity_attn = self.entity_attn(entity_q, text_tokens, text_tokens, key_padding_mask=text_padding_mask)
        comp_tokens, comp_attn = self.comp_attn(comp_q, text_tokens, text_tokens, key_padding_mask=text_padding_mask)

        return {
            "scene": self.norm_scene(scene_tokens),
            "entity": self.norm_entity(entity_tokens),
            "composition": self.norm_comp(comp_tokens),
            "attn": {
                "scene_text": scene_attn,
                "entity_text": entity_attn,
                "composition_text": comp_attn,
            },
        }


class SparseRelationAggregator(nn.Module):
    """Aggregate relation-aware node context from sparse Top-K graph edges."""

    def __init__(self, embed_dim, relation_names=("adj", "sem", "co", "freq")):
        super().__init__()
        self.relation_names = tuple(relation_names)
        self.relation_proj = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            for r in self.relation_names
        })
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * len(self.relation_names), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _gather_neighbors(x, idx):
        # x: [B, N, D], idx: [B, N, K] -> [B, N, K, D]
        B, N, D = x.shape
        K = idx.shape[-1]
        idx = idx.unsqueeze(-1).expand(B, N, K, D)
        x_expand = x.unsqueeze(1).expand(B, N, N, D)
        return torch.gather(x_expand, dim=2, index=idx)

    def forward(self, nodes, relation_graphs):
        relation_contexts = []
        for relation in self.relation_names:
            if relation not in relation_graphs:
                raise KeyError(f"relation_graphs must contain relation '{relation}'.")

            idx, weight = relation_graphs[relation]
            if idx.dim() != 3 or weight.dim() != 3:
                raise ValueError(f"relation '{relation}' must be a tuple of [B, N, K] tensors.")
            if idx.shape[:2] != nodes.shape[:2] or weight.shape[:2] != nodes.shape[:2]:
                raise ValueError(
                    f"relation '{relation}' has node size {idx.shape[:2]}, "
                    f"but nodes have size {nodes.shape[:2]}."
                )

            neigh = self._gather_neighbors(nodes, idx)
            attn = torch.softmax(weight, dim=-1).unsqueeze(-1)
            ctx = torch.sum(attn * neigh, dim=2)
            ctx = self.relation_proj[relation](ctx)
            relation_contexts.append(ctx)

        return self.fuse(torch.cat(relation_contexts, dim=-1))


class RelationAwareCompositionBuilder(nn.Module):
    """Build composition-level visual evidence with graph relations."""

    def __init__(self, embed_dim, freq_dim=4, relation_names=("adj", "sem", "co", "freq")):
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.rel_agg = SparseRelationAggregator(embed_dim, relation_names)
        self.comp_fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_nodes, freq_desc, relation_graphs):
        freq_embed = self.freq_proj(freq_desc)
        rel_context = self.rel_agg(visual_nodes, relation_graphs)
        comp_nodes = self.comp_fuse(torch.cat([visual_nodes, freq_embed, rel_context], dim=-1))
        return self.out_norm(visual_nodes + comp_nodes)


class FrequencyGuidedStructureAwareMultiGranularityAlignment(nn.Module):
    """F-SAMGA module.

    The module performs image-text alignment at three levels:
        1) Scene-level: global scene structure <-> sentence-level semantics.
        2) Entity-level: object/land-cover evidence <-> noun/object semantics.
        3) Composition-level: graph relation evidence <-> phrases/relations.

    This implementation follows the new design directly and does not preserve the
    old GlobalAlignment/LocalAlignment/LatentSemanticAlignment API.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_entity_queries=16,
        num_comp_queries=16,
        dropout=0.1,
        ot_eps=0.07,
        ot_iter=50,
        weights=(1.0, 1.0, 1.0),
        freq_dim=4,
        relation_names=("adj", "sem", "co", "freq"),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.weights = weights
        self.relation_names = tuple(relation_names)

        self.visual_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_extractor = TextGranularityExtractor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_entity_queries=num_entity_queries,
            num_comp_queries=num_comp_queries,
            dropout=dropout,
        )

        # Frequency-guided visual node adapters.
        self.scene_freq_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.entity_freq_proj = nn.Sequential(nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.comp_builder = RelationAwareCompositionBuilder(embed_dim, freq_dim=freq_dim, relation_names=relation_names)

        # Scene branch: LL-guided attention pooling over visual evidence.
        self.scene_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.scene_norm = nn.LayerNorm(embed_dim)

        # Entity branch: text-conditioned retrieval of high-frequency visual evidence.
        self.entity_visual_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.entity_norm = nn.LayerNorm(embed_dim)

        # Composition branch: text-conditioned retrieval of relation-aware visual evidence.
        self.comp_visual_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.comp_norm = nn.LayerNorm(embed_dim)

        self.ot = EntropicOT(eps=ot_eps, max_iter=ot_iter)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    @staticmethod
    def _cosine_cost(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sim = torch.bmm(x, y.transpose(1, 2))
        return 1.0 - sim

    def _temp(self):
        return self.temperature.clamp(min=1e-4)

    def _scene_alignment(self, visual_nodes, freq_desc, text_scene):
        # LL guides scene-level visual evidence selection.
        ll = freq_desc[:, :, 0:1]
        scene_nodes = self.scene_norm(visual_nodes + self.scene_freq_proj(ll))
        scene_weight = torch.softmax(self.scene_gate(scene_nodes), dim=1)
        visual_scene = torch.sum(scene_weight * scene_nodes, dim=1)

        text_scene = text_scene.squeeze(1)
        visual_scene = F.normalize(visual_scene, dim=-1)
        text_scene = F.normalize(text_scene, dim=-1)
        score = torch.sum(visual_scene * text_scene, dim=-1) / self._temp()
        loss = 1.0 - torch.tanh(score).mean()
        return loss, score, scene_weight.squeeze(-1), visual_scene

    def _entity_alignment(self, visual_nodes, freq_desc, text_entity):
        # LH/HL/HH guide entity-level evidence such as boundaries and small objects.
        high_freq = freq_desc[:, :, 1:4]
        entity_nodes = self.entity_norm(visual_nodes + self.entity_freq_proj(high_freq))
        visual_entity, entity_attn = self.entity_visual_attn(text_entity, entity_nodes, entity_nodes)
        visual_entity = self.entity_norm(visual_entity)

        cost = self._cosine_cost(visual_entity, text_entity)
        dist, plan = self.ot(cost)
        score = -dist / self._temp()
        loss = dist.mean()
        return loss, score, entity_attn, plan, visual_entity

    def _composition_alignment(self, visual_nodes, freq_desc, relation_graphs, text_comp):
        # Graph relations + four DWT sub-bands guide composition-level alignment.
        comp_nodes = self.comp_builder(visual_nodes, freq_desc, relation_graphs)
        visual_comp, comp_attn = self.comp_visual_attn(text_comp, comp_nodes, comp_nodes)
        visual_comp = self.comp_norm(visual_comp)

        cost = self._cosine_cost(visual_comp, text_comp)
        dist, plan = self.ot(cost)
        score = -dist / self._temp()
        loss = dist.mean()
        return loss, score, comp_attn, plan, visual_comp, comp_nodes

    def forward(self, visual_evidence, freq_desc, relation_graphs, text_tokens, text_padding_mask=None, return_details=False):
        """Run F-SAMGA.

        Args:
            visual_evidence: Tensor[B, N, D], graph-enhanced visual evidence.
            freq_desc: Tensor[B, N, 4], node-level [LL, LH, HL, HH] descriptors.
            relation_graphs: Dict[str, Tuple[index, weight]], sparse graph edges.
            text_tokens: Tensor[B, M, D], token-level text features.
            text_padding_mask: optional BoolTensor[B, M], True means padding token.
            return_details: if True, return intermediate scores and maps.
        """
        if visual_evidence.dim() != 3:
            raise ValueError("visual_evidence must have shape [B, N, D].")
        if freq_desc.dim() != 3 or freq_desc.shape[-1] != 4:
            raise ValueError("freq_desc must have shape [B, N, 4] in the order [LL, LH, HL, HH].")
        if text_tokens.dim() != 3:
            raise ValueError("text_tokens must have shape [B, M, D].")
        if visual_evidence.shape[0] != text_tokens.shape[0] or visual_evidence.shape[0] != freq_desc.shape[0]:
            raise ValueError("visual_evidence, freq_desc, and text_tokens must have the same batch size.")
        if visual_evidence.shape[1] != freq_desc.shape[1]:
            raise ValueError("visual_evidence and freq_desc must have the same number of nodes.")

        visual_nodes = self.visual_proj(visual_evidence)
        text_nodes = self.text_proj(text_tokens)
        text_parts = self.text_extractor(text_nodes, text_padding_mask=text_padding_mask)

        scene_loss, scene_score, scene_weight, visual_scene = self._scene_alignment(
            visual_nodes, freq_desc, text_parts["scene"]
        )
        entity_loss, entity_score, entity_attn, entity_plan, visual_entity = self._entity_alignment(
            visual_nodes, freq_desc, text_parts["entity"]
        )
        comp_loss, comp_score, comp_attn, comp_plan, visual_comp, comp_nodes = self._composition_alignment(
            visual_nodes, freq_desc, relation_graphs, text_parts["composition"]
        )

        total_loss = (
            self.weights[0] * scene_loss +
            self.weights[1] * entity_loss +
            self.weights[2] * comp_loss
        )
        total_score = (
            self.weights[0] * scene_score +
            self.weights[1] * entity_score +
            self.weights[2] * comp_score
        )

        if not return_details:
            return total_loss

        details = {
            "total_loss": total_loss,
            "scene_loss": scene_loss,
            "entity_loss": entity_loss,
            "composition_loss": comp_loss,
            "total_score": total_score,
            "scene_score": scene_score,
            "entity_score": entity_score,
            "composition_score": comp_score,
            "scene_weight": scene_weight,
            "entity_attention": entity_attn,
            "composition_attention": comp_attn,
            "entity_transport_plan": entity_plan,
            "composition_transport_plan": comp_plan,
            "visual_scene": visual_scene,
            "visual_entity": visual_entity,
            "visual_composition": visual_comp,
            "composition_nodes": comp_nodes,
            "text_scene": text_parts["scene"],
            "text_entity": text_parts["entity"],
            "text_composition": text_parts["composition"],
            "text_attention": text_parts["attn"],
        }
        return total_loss, details


# Alias used in the paper.
F_SAMGA = FrequencyGuidedStructureAwareMultiGranularityAlignment
