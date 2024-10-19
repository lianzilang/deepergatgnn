import sys
sys.path.append('/home/aistudio/work/paddle_project/utils')
from utils import paddle_aux
import paddle
import paddle.nn.functional as F
# import torch_geometric
# from torch_geometric.nn import Set2Set, global_mean_pool, global_add_pool, global_max_pool, GCNConv
# from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter


class GCN(paddle.nn.Layer):

    def __init__(self, data, dim1=64, dim2=64, pre_fc_count=1, gc_count=3,
        post_fc_count=1, pool='global_mean_pool', pool_order='early',
        batch_norm='True', batch_track_stats='True', act='relu',
        dropout_rate=0.0, **kwargs):
        super(GCN, self).__init__()
        if batch_track_stats == 'False':
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        assert gc_count > 0, 'Need at least 1 GC layer'
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])
        if pre_fc_count > 0:
            self.pre_lin_list = paddle.nn.LayerList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = paddle.nn.Linear(in_features=data.num_features,
                        out_features=dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = paddle.nn.Linear(in_features=dim1, out_features=dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = paddle.nn.LayerList()
        self.conv_list = paddle.nn.LayerList()
        self.bn_list = paddle.nn.LayerList()
        for i in range(gc_count):
            conv = GCNConv(gc_dim, gc_dim, improved=True, add_self_loops=False)
            self.conv_list.append(conv)
            if self.batch_norm == 'True':
                bn = paddle.nn.BatchNorm1D(num_features=gc_dim)
                self.bn_list.append(bn)
        if post_fc_count > 0:
            self.post_lin_list = paddle.nn.LayerList()
            for i in range(post_fc_count):
                if i == 0:
                    if self.pool_order == 'early' and self.pool == 'set2set':
                        lin = paddle.nn.Linear(in_features=post_fc_dim * 2,
                            out_features=dim2)
                    else:
                        lin = paddle.nn.Linear(in_features=post_fc_dim,
                            out_features=dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = paddle.nn.Linear(in_features=dim2, out_features=dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = paddle.nn.Linear(in_features=dim2, out_features=
                output_dim)
        elif post_fc_count == 0:
            self.post_lin_list = paddle.nn.LayerList()
            if self.pool_order == 'early' and self.pool == 'set2set':
                self.lin_out = paddle.nn.Linear(in_features=post_fc_dim * 2,
                    out_features=output_dim)
            else:
                self.lin_out = paddle.nn.Linear(in_features=post_fc_dim,
                    out_features=output_dim)
        if self.pool_order == 'early' and self.pool == 'set2set':
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == 'late' and self.pool == 'set2set':
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1
                )
            self.lin_out_2 = paddle.nn.Linear(in_features=output_dim * 2,
                out_features=output_dim)

    def forward(self, data):
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == 'True':
                    out = self.conv_list[i](data.x, data.edge_index, data.
                        edge_weight)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, data.edge_index, data.
                        edge_weight)
            elif self.batch_norm == 'True':
                out = self.conv_list[i](out, data.edge_index, data.edge_weight)
                out = self.bn_list[i](out)
            else:
                out = self.conv_list[i](out, data.edge_index, data.edge_weight)
            out = getattr(F, self.act)(out)
            out = paddle.nn.functional.dropout(x=out, p=self.dropout_rate,
                training=self.training)
        if self.pool_order == 'early':
            if self.pool == 'set2set':
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
        elif self.pool_order == 'late':
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == 'set2set':
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
        if tuple(out.shape)[1] == 1:
            return out.view(-1)
        else:
            return out
