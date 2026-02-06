from math import sqrt
import torch
from torch import Tensor
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.version import debug
from torch_geometric.nn import MessagePassing
from dig.xgraph.method.base_explainer import ExplainerBase
from typing import Union


class ProvX(ExplainerBase):
    r"""
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        explain_graph (bool, optional): Whether to explain graph classification model
            (default: :obj:`False`)
        indirect_graph_symmetric_weights (bool, optional): If `True`, then the explainer
            will first realize whether this graph input has indirect edges,
            then makes its edge weights symmetric. (default: :obj:`False`)
        alpha (float, optional): Balancing coefficient for the two loss parts.
            (default: :obj:`0.1`)
        L1_dist (bool, optional): Whether to use L1 distance for edge_dist_loss.
            (default: :obj:`False`)
        solidification_factor (float, optional): Factor for the solidification penalty.
            If 0, this stage is skipped. (default: :obj:`0.0`)
        solidification_stage_start_ratio (float, optional): Ratio of epochs after which
            to start the solidification stage. (default: :obj:`0.5`)
        confident_threshold_low (float, optional): Lower threshold for considering an edge mask
            value "confident" (close to 0). (default: :obj:`0.1`)
        confident_threshold_high (float, optional): Upper threshold for considering an edge mask
            value "confident" (close to 1). (default: :obj:`0.9`)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        alpha: float = 0.1,
        explain_graph: bool = False,
        indirect_graph_symmetric_weights: bool = False,
        L1_dist: bool = False,
        solidification_factor: float = 0.0,
        solidification_stage_start_ratio: float = 0.5,
        confident_threshold_low: float = 0.1,
        confident_threshold_high: float = 0.9,
    ):
        super(ProvX, self).__init__(model, epochs, lr, explain_graph)
        self.alpha = alpha
        self.L1_dist = L1_dist
        self._symmetric_edge_mask_indirect_graph: bool = indirect_graph_symmetric_weights

        self.solidification_factor = solidification_factor
        self.solidification_stage_start_ratio = solidification_stage_start_ratio
        self.confident_threshold_low = confident_threshold_low
        self.confident_threshold_high = confident_threshold_high

    def __set_masks__(self, x: Tensor, edge_index: Tensor, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(
            torch.randn(F, requires_grad=True, device=self.device) * 0.1
        )

        if E == 0:
            self.edge_mask = torch.nn.Parameter(
                torch.empty(0, requires_grad=True, device=self.device)
            )
        else:
            if N > 0:
                std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
            else:
                std = 0.02
            self.edge_mask = torch.nn.Parameter(
                torch.randn(E, requires_grad=True, device=self.device) * std
            )

        loop_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        if E > 0:
            loop_mask = edge_index[0] != edge_index[1]

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.node_feat_mask = None
        self.edge_mask = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        relu = torch.nn.ReLU()

        pred_loss_val = 0
        if self.explain_graph:
            pred_loss_val = relu(
                raw_preds[
                    :,
                    x_label.item()
                    if isinstance(x_label, Tensor) and x_label.numel() == 1
                    else x_label,
                ]
            ).sum()
        else:
            pred_loss_val = relu(
                raw_preds[
                    self.node_idx,
                    x_label.item()
                    if isinstance(x_label, Tensor) and x_label.numel() == 1
                    else x_label,
                ]
            ).sum()

        edge_dist_loss_val = 0
        if self.edge_mask is not None and self.edge_mask.numel() > 0:
            m = self.edge_mask.sigmoid()
            if self.L1_dist:
                edge_dist_loss_val = torch.linalg.norm(1 - m, ord=1)
            else:
                edge_dist_loss_val = torch.nn.functional.binary_cross_entropy(
                    m, torch.ones_like(m).to(m.device)
                )

        loss = self.alpha * pred_loss_val + (1 - self.alpha) * edge_dist_loss_val
        return loss

    def gnn_explainer_alg(
        self,
        x: Tensor,
        edge_index: Tensor,
        ex_label: Tensor,
        mask_features: bool = False,
        **kwargs,
    ) -> Tensor:
        self.to(x.device)
        self.mask_features = mask_features

        params_to_optimize = []
        if self.edge_mask is not None and self.edge_mask.requires_grad and self.edge_mask.numel() > 0:
            params_to_optimize.append(self.edge_mask)
        if self.mask_features and self.node_feat_mask is not None and self.node_feat_mask.requires_grad:
            params_to_optimize.append(self.node_feat_mask)

        if not params_to_optimize:
            return (
                self.edge_mask.data
                if self.edge_mask is not None
                else torch.empty(0, device=x.device)
            )

        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)

        m_at_stage_start = None
        confident_indices_low = None
        confident_indices_high = None

        solidification_start_epoch = 0
        if self.epochs > 0:
            solidification_start_epoch = max(
                1, int(self.epochs * self.solidification_stage_start_ratio)
            )

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            if (
                epoch == solidification_start_epoch
                and self.solidification_factor > 0.0
                and self.edge_mask is not None
                and self.edge_mask.numel() > 0
            ):
                with torch.no_grad():
                    m_activated_snapshot = self.edge_mask.sigmoid()
                    m_at_stage_start = m_activated_snapshot.detach().clone()

                    confident_indices_low = (
                        m_at_stage_start < self.confident_threshold_low
                    ).nonzero(as_tuple=True)[0]
                    confident_indices_high = (
                        m_at_stage_start > self.confident_threshold_high
                    ).nonzero(as_tuple=True)[0]

                    if debug:
                        print(
                            f"Epoch {epoch}/{self.epochs}: Solidification stage started. "
                            f"Found {confident_indices_low.numel()} low-confidence edges and "
                            f"{confident_indices_high.numel()} high-confidence edges to solidify."
                        )

            h = x
            if self.mask_features and self.node_feat_mask is not None:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()

            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if (
                epoch > solidification_start_epoch
                and self.solidification_factor > 0.0
                and m_at_stage_start is not None
                and self.edge_mask is not None
                and self.edge_mask.numel() > 0
            ):
                current_m_activated = self.edge_mask.sigmoid()
                current_solidification_penalty = torch.tensor(0.0, device=self.device)

                if confident_indices_low is not None and confident_indices_low.numel() > 0:
                    deviation_low = (
                        current_m_activated[confident_indices_low]
                        - m_at_stage_start[confident_indices_low]
                    )
                    current_solidification_penalty += deviation_low.pow(2).sum()

                if (
                    confident_indices_high is not None
                    and confident_indices_high.numel() > 0
                ):
                    deviation_high = (
                        current_m_activated[confident_indices_high]
                        - m_at_stage_start[confident_indices_high]
                    )
                    current_solidification_penalty += deviation_high.pow(2).sum()

                loss += self.solidification_factor * current_solidification_penalty

            loss.backward()

            params_to_clip_grads = []
            if self.node_feat_mask is not None and self.node_feat_mask.requires_grad:
                params_to_clip_grads.append(self.node_feat_mask)
            if self.edge_mask is not None and self.edge_mask.requires_grad and self.edge_mask.numel() > 0:
                params_to_clip_grads.append(self.edge_mask)

            if params_to_clip_grads:
                torch.nn.utils.clip_grad_value_(params_to_clip_grads, clip_value=2.0)

            optimizer.step()

            if epoch % 20 == 0 and debug:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item()}")

        return (
            self.edge_mask.data
            if self.edge_mask is not None
            else torch.empty(0, device=x.device)
        )

    def forward(self, x, edge_index, mask_features=False, target_label=None, **kwargs):
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        effective_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        x_for_alg = x
        ei_for_alg = effective_edge_index

        if not self.explain_graph:
            self.node_idx = node_idx = kwargs.get("node_idx")
            assert node_idx is not None, "An node explanation needs kwarg node_idx, but got None."
            pass

        output_edge_masks = []
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            with torch.no_grad():
                temp_logits = self.model(
                    x, effective_edge_index, **kwargs.get("model_kwargs", {})
                )
                num_classes = temp_logits.size(-1)

        labels_to_explain = []
        if target_label is not None:
            if isinstance(target_label, torch.Tensor):
                labels_to_explain.append(target_label)
            else:
                labels_to_explain.append(torch.tensor([target_label], device=self.device))
        else:
            labels_to_explain = [
                torch.tensor([i], device=self.device) for i in range(num_classes)
            ]

        for current_ex_label_tensor in labels_to_explain:
            self.__clear_masks__()
            self.__set_masks__(x_for_alg, ei_for_alg)

            model_fwd_kwargs = kwargs.get("model_kwargs", kwargs)

            edge_mask_data = self.gnn_explainer_alg(
                x_for_alg,
                ei_for_alg,
                current_ex_label_tensor,
                mask_features=mask_features,
                **model_fwd_kwargs,
            )
            output_edge_masks.append(
                edge_mask_data.sigmoid()
                if edge_mask_data.numel() > 0
                else edge_mask_data
            )

        hard_edge_masks = [
            self.control_sparsity(mask, sparsity=kwargs.get("sparsity"))
            for mask in output_edge_masks
        ]

        related_preds = self.eval_related_pred(x_for_alg, ei_for_alg, hard_edge_masks, **kwargs)

        self.__clear_masks__()
        return output_edge_masks, hard_edge_masks, related_preds, ei_for_alg

    def __repr__(self):
        return f"{self.__class__.__name__}()"

