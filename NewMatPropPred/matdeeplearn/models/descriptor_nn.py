import sys
sys.path.append('/home/aistudio/work/paddle_project/utils')
import paddle_aux
import paddle
from torch_geometric.nn import Set2Set, global_mean_pool, global_add_pool, global_max_pool


class SM(paddle.nn.Layer):

    def __init__(self, data, dim1=64, fc_count=1, **kwargs):
        super(SM, self).__init__()
        self.lin1 = paddle.nn.Linear(in_features=data[0].extra_features_SM.
            shape[1], out_features=dim1)
        self.lin_list = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=dim1, out_features=dim1) for i in range(fc_count)])
        self.lin2 = paddle.nn.Linear(in_features=dim1, out_features=1)

    def forward(self, data):
        out = paddle.nn.functional.relu(x=self.lin1(data.extra_features_SM))
        for layer in self.lin_list:
            out = paddle.nn.functional.relu(x=layer(out))
        out = self.lin2(out)
        if tuple(out.shape)[1] == 1:
            return out.view(-1)
        else:
            return out


class SOAP(paddle.nn.Layer):

    def __init__(self, data, dim1, fc_count, **kwargs):
        super(SOAP, self).__init__()
        self.lin1 = paddle.nn.Linear(in_features=data[0].
            extra_features_SOAP.shape[1], out_features=dim1)
        self.lin_list = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=dim1, out_features=dim1) for i in range(fc_count)])
        self.lin2 = paddle.nn.Linear(in_features=dim1, out_features=1)

    def forward(self, data):
        out = paddle.nn.functional.relu(x=self.lin1(data.extra_features_SOAP))
        for layer in self.lin_list:
            out = paddle.nn.functional.relu(x=layer(out))
        out = self.lin2(out)
        if tuple(out.shape)[1] == 1:
            return out.view(-1)
        else:
            return out
