import sys
sys.path.append('/home/aistudio/work/paddle_project/utils')
from utils import paddle_aux
import paddle
import paddle.nn.functional as F
import torch_geometric
from torch_geometric.nn import Set2Set, global_mean_pool, global_add_pool, global_max_pool, MetaLayer, DiffGroupNorm
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter


class Megnet_EdgeModel(paddle.nn.Layer):

    def __init__(self, dim, act, batch_norm, batch_track_stats,
        dropout_rate, fc_layers=2):
        super(Megnet_EdgeModel, self).__init__()
        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == 'False':
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.edge_mlp = paddle.nn.LayerList()
        self.bn_list = paddle.nn.LayerList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = paddle.nn.Linear(in_features=dim * 4, out_features=dim)
                self.edge_mlp.append(lin)
            else:
                lin = paddle.nn.Linear(in_features=dim, out_features=dim)
                self.edge_mlp.append(lin)
            if self.batch_norm == 'True':
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.
                    batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, src, dest, edge_attr, u, batch):
        comb = paddle.concat(x=[src, dest, edge_attr, u[batch]], axis=1)
        for i in range(0, len(self.edge_mlp)):
            if i == 0:
                out = self.edge_mlp[i](comb)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
            else:
                out = self.edge_mlp[i](out)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.add(x=out, y=paddle.to_tensor(prev_out))
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
        return out


class Megnet_NodeModel(paddle.nn.Layer):

    def __init__(self, dim, act, batch_norm, batch_track_stats,
        dropout_rate, fc_layers=2):
        super(Megnet_NodeModel, self).__init__()
        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == 'False':
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.node_mlp = paddle.nn.LayerList()
        self.bn_list = paddle.nn.LayerList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = paddle.nn.Linear(in_features=dim * 3, out_features=dim)
                self.node_mlp.append(lin)
            else:
                lin = paddle.nn.Linear(in_features=dim, out_features=dim)
                self.node_mlp.append(lin)
            if self.batch_norm == 'True':
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.
                    batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, edge_index, edge_attr, u, batch):
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = paddle.concat(x=[x, v_e, u[batch]], axis=1)
        for i in range(0, len(self.node_mlp)):
            if i == 0:
                out = self.node_mlp[i](comb)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
            else:
                out = self.node_mlp[i](out)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.add(x=out, y=paddle.to_tensor(prev_out))
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
        return out


class Megnet_GlobalModel(paddle.nn.Layer):

    def __init__(self, dim, act, batch_norm, batch_track_stats,
        dropout_rate, fc_layers=2):
        super(Megnet_GlobalModel, self).__init__()
        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == 'False':
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.global_mlp = paddle.nn.LayerList()
        self.bn_list = paddle.nn.LayerList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = paddle.nn.Linear(in_features=dim * 3, out_features=dim)
                self.global_mlp.append(lin)
            else:
                lin = paddle.nn.Linear(in_features=dim, out_features=dim)
                self.global_mlp.append(lin)
            if self.batch_norm == 'True':
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.
                    batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = paddle.concat(x=[u_e, u_v, u], axis=1)
        for i in range(0, len(self.global_mlp)):
            if i == 0:
                out = self.global_mlp[i](comb)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
            else:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)
                if self.batch_norm == 'True':
                    out = self.bn_list[i](out)
                out = paddle.add(x=out, y=paddle.to_tensor(prev_out))
                out = paddle.nn.functional.dropout(x=out, p=self.
                    dropout_rate, training=self.training)
                prev_out = out
        return out


class SUPER_MEGNet(paddle.nn.Layer):

    def __init__(self, data, dim1=64, dim2=64, dim3=64, pre_fc_count=1,
        gc_count=3, gc_fc_count=2, post_fc_count=1, pool='global_mean_pool',
        pool_order='early', batch_norm='True', batch_track_stats='True',
        act='relu', dropout_rate=0.0, **kwargs):
        super(SUPER_MEGNet, self).__init__()
        if batch_track_stats == 'False':
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.pool = pool
        if pool == 'global_mean_pool':
            self.pool_reduce = 'mean'
        elif pool == 'global_max_pool':
            self.pool_reduce = 'max'
        elif pool == 'global_sum_pool':
            self.pool_reduce = 'sum'
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        assert gc_count > 0, 'Need at least 1 GC layer'
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        post_fc_dim = dim3
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
        self.e_embed_list = paddle.nn.LayerList()
        self.x_embed_list = paddle.nn.LayerList()
        self.u_embed_list = paddle.nn.LayerList()
        self.conv_list = paddle.nn.LayerList()
        self.bn_list = paddle.nn.LayerList()
        for i in range(gc_count):
            if i == 0:
                e_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =data.num_edge_features, out_features=dim3), paddle.nn.
                    ReLU(), paddle.nn.Linear(in_features=dim3, out_features
                    =dim3), paddle.nn.ReLU())
                x_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =gc_dim, out_features=dim3), paddle.nn.ReLU(), paddle.
                    nn.Linear(in_features=dim3, out_features=dim3), paddle.
                    nn.ReLU())
                u_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =data[0].u.shape[1], out_features=dim3), paddle.nn.ReLU
                    (), paddle.nn.Linear(in_features=dim3, out_features=
                    dim3), paddle.nn.ReLU())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(MetaLayer(Megnet_EdgeModel(dim3, self
                    .act, self.batch_norm, self.batch_track_stats, self.
                    dropout_rate, gc_fc_count), Megnet_NodeModel(dim3, self
                    .act, self.batch_norm, self.batch_track_stats, self.
                    dropout_rate, gc_fc_count), Megnet_GlobalModel(dim3,
                    self.act, self.batch_norm, self.batch_track_stats, self
                    .dropout_rate, gc_fc_count)))
            elif i > 0:
                e_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =dim3, out_features=dim3), paddle.nn.ReLU(), paddle.nn.
                    Linear(in_features=dim3, out_features=dim3), paddle.nn.
                    ReLU())
                x_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =dim3, out_features=dim3), paddle.nn.ReLU(), paddle.nn.
                    Linear(in_features=dim3, out_features=dim3), paddle.nn.
                    ReLU())
                u_embed = paddle.nn.Sequential(paddle.nn.Linear(in_features
                    =dim3, out_features=dim3), paddle.nn.ReLU(), paddle.nn.
                    Linear(in_features=dim3, out_features=dim3), paddle.nn.
                    ReLU())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(MetaLayer(Megnet_EdgeModel(dim3, self
                    .act, self.batch_norm, self.batch_track_stats, self.
                    dropout_rate, gc_fc_count), Megnet_NodeModel(dim3, self
                    .act, self.batch_norm, self.batch_track_stats, self.
                    dropout_rate, gc_fc_count), Megnet_GlobalModel(dim3,
                    self.act, self.batch_norm, self.batch_track_stats, self
                    .dropout_rate, gc_fc_count)))
        if post_fc_count > 0:
            self.post_lin_list = paddle.nn.LayerList()
            for i in range(post_fc_count):
                if i == 0:
                    if self.pool_order == 'early' and self.pool == 'set2set':
                        lin = paddle.nn.Linear(in_features=post_fc_dim * 5,
                            out_features=dim2)
                    elif self.pool_order == 'early' and self.pool != 'set2set':
                        lin = paddle.nn.Linear(in_features=post_fc_dim * 3,
                            out_features=dim2)
                    elif self.pool_order == 'late':
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
                self.lin_out = paddle.nn.Linear(in_features=post_fc_dim * 5,
                    out_features=output_dim)
            elif self.pool_order == 'early' and self.pool != 'set2set':
                self.lin_out = paddle.nn.Linear(in_features=post_fc_dim * 3,
                    out_features=output_dim)
            else:
                self.lin_out = paddle.nn.Linear(in_features=post_fc_dim,
                    out_features=output_dim)
        if self.pool_order == 'early' and self.pool == 'set2set':
            self.set2set_x = Set2Set(post_fc_dim, processing_steps=3)
            self.set2set_e = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == 'late' and self.pool == 'set2set':
            self.set2set_x = Set2Set(output_dim, processing_steps=3,
                num_layers=1)
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
            if i == 0:
                if len(self.pre_lin_list) == 0:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](data.x)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](x_temp, data.
                        edge_index, e_temp, u_temp, data.batch)
                    x = paddle.add(x=x_out, y=paddle.to_tensor(x_temp))
                    e = paddle.add(x=e_out, y=paddle.to_tensor(e_temp))
                    u = paddle.add(x=u_out, y=paddle.to_tensor(u_temp))
                else:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](out)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](x_temp, data.
                        edge_index, e_temp, u_temp, data.batch)
                    x = paddle.add(x=x_out, y=paddle.to_tensor(x_temp))
                    e = paddle.add(x=e_out, y=paddle.to_tensor(e_temp))
                    u = paddle.add(x=u_out, y=paddle.to_tensor(u_temp))
                prev_x = x
                prev_e = e
                prev_u = u
            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](x_temp, data.
                    edge_index, e_temp, u_temp, data.batch)
                x = paddle.add(x=x_out, y=paddle.to_tensor(x))
                e = paddle.add(x=e_out, y=paddle.to_tensor(e))
                u = paddle.add(x=u_out, y=paddle.to_tensor(u))
                x = paddle.add(x=x, y=paddle.to_tensor(prev_x))
                e = paddle.add(x=e, y=paddle.to_tensor(prev_e))
                u = paddle.add(x=u, y=paddle.to_tensor(prev_u))
                prev_x = x
                prev_e = e
                prev_u = u
        if self.pool_order == 'early':
            if self.pool == 'set2set':
                x_pool = self.set2set_x(x, data.batch)
                e = scatter(e, data.edge_index[0, :], dim=0, reduce='mean')
                e_pool = self.set2set_e(e, data.batch)
                out = paddle.concat(x=[x_pool, e_pool, u], axis=1)
            else:
                x_pool = scatter(x, data.batch, dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e, data.edge_index[0, :], dim=0, reduce=
                    self.pool_reduce)
                e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.
                    pool_reduce)
                out = paddle.concat(x=[x_pool, e_pool, u], axis=1)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
        elif self.pool_order == 'late':
            out = x
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == 'set2set':
                out = self.set2set_x(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
        if tuple(out.shape)[1] == 1:
            return out.view(-1)
        else:
            return out
