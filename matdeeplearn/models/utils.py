import paddle


def model_summary(model):
    model_params_list = list(model.named_parameters())
    print(
        '--------------------------------------------------------------------------'
        )
    line_new = '{:>30}  {:>20} {:>20}'.format('Layer.Parameter',
        'Param Tensor Shape', 'Param #')
    print(line_new)
    print(
        '--------------------------------------------------------------------------'
        )
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(tuple(elem[1].shape))
        p_count = paddle.to_tensor(data=tuple(elem[1].shape)).prod().item()
        line_new = '{:>30}  {:>20} {:>20}'.format(p_name, str(p_shape), str
            (p_count))
        print(line_new)
    print(
        '--------------------------------------------------------------------------'
        )
    total_params = sum([param.size for param in model.parameters()])
    print('Total params:', total_params)
    num_trainable_params = sum(p.size for p in model.parameters() if not p.
        stop_gradient)
    print('Trainable params:', num_trainable_params)
    print('Non-trainable params:', total_params - num_trainable_params)
