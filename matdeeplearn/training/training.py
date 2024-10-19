import os
import paddle
import csv
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
# from torch_geometric.data import DataLoader, Dataset
# from torch_geometric.nn import DataParallel
# import torch_geometric.transforms as T
from matdeeplearn import models
import matdeeplearn.process as process
import matdeeplearn.training as training
from matdeeplearn.models.utils import model_summary


def train(model, optimizer, loader, loss_method, rank):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        optimizer.clear_gradients(set_to_zero=False)
        output = model(data)
        loss = getattr(F, loss_method)(output, data.y)
        loss.backward()
        loss_all += loss.detach() * output.shape[0]
        optimizer.step()
        count = count + output.shape[0]
    loss_all = loss_all / count
    return loss_all


def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with paddle.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.shape[0]
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in
                        sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                else:
                    ids_temp = [item for sublist in data.structure_id for
                        item in sublist]
                    ids_temp = [item for sublist in ids_temp for item in
                        sublist]
                    ids = ids + ids_temp
                    predict = np.concatenate((predict, output.data.cpu().
                        numpy()), axis=0)
                    target = np.concatenate((target, data.y.cpu().numpy()),
                        axis=0)
            count = count + output.shape[0]
    loss_all = loss_all / count
    if out == True:
        test_out = np.column_stack((ids, target, predict))
        return loss_all, test_out
    elif out == False:
        return loss_all


def trainer(rank, world_size, model, optimizer, scheduler, loss,
    train_loader, val_loader, train_sampler, epochs, verbosity, filename=
    'my_model_temp.pth'):
    train_error = val_error = test_error = epoch_time = float('NaN')
    train_start = time.time()
    best_val_error = 10000000000.0
    model_best = model
    for epoch in range(1, epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        if rank not in ('cpu', 'cuda'):
            train_sampler.set_epoch(epoch)
        train_error = train(model, optimizer, train_loader, loss, rank=rank)
        if rank not in ('cpu', 'cuda'):
            paddle.distributed.reduce(tensor=train_error, dst=0)
            train_error = train_error / world_size
        if rank not in ('cpu', 'cuda'):
            paddle.distributed.barrier()
        if val_loader != None and rank in (0, 'cpu', 'cuda'):
            if rank not in ('cpu', 'cuda'):
                val_error = evaluate(val_loader, model.module, loss, rank=
                    rank, out=False)
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank,
                    out=False)
        epoch_time = time.time() - train_start
        train_start = time.time()
        if val_loader != None and rank in (0, 'cpu', 'cuda'):
            if val_error == float('NaN') or val_error < best_val_error:
                if rank not in ('cpu', 'cuda'):
                    model_best = copy.deepcopy(model.module)
                    paddle.save(obj={'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'full_model': model}, path=filename)
                else:
                    model_best = copy.deepcopy(model)
                    paddle.save(obj={'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'full_model': model}, path=filename)
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, 'cpu', 'cuda'):
            if rank not in ('cpu', 'cuda'):
                model_best = copy.deepcopy(model.module)
                paddle.save(obj={'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'full_model': model}, path=filename)
            else:
                model_best = copy.deepcopy(model)
                paddle.save(obj={'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'full_model': model}, path=filename)
        scheduler.step(train_error)
        if epoch % verbosity == 0:
            if rank in (0, 'cpu', 'cuda'):
                print(
                    'Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}'
                    .format(epoch, lr, train_error, val_error, epoch_time))
    if rank not in ('cpu', 'cuda'):
        paddle.distributed.barrier()
    return model_best


def write_results(output, filename):
    shape = tuple(output.shape)
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(['ids'] + ['target'] * int((shape[1] - 1
                    ) / 2) + ['prediction'] * int((shape[1] - 1) / 2))
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


def ddp_setup(rank, world_size):
    if rank in ('cpu', 'cuda'):
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if platform.system() == 'Windows':
        paddle.distributed.init_parallel_env()
    else:
        paddle.distributed.init_parallel_env()
    True = False
    False = True


def model_setup(rank, model_name, model_params, dataset, load_model=False,
    model_path=None, print_model=True):
    model = getattr(models, model_name)(data=dataset, **model_params if 
        model_params is not None else {}).to(rank)
    if load_model == 'True':
        assert os.path.exists(model_path), 'Saved model not found'
        if str(rank) in 'cpu':
            saved = paddle.load(path=str(model_path))
        else:
            saved = paddle.load(path=str(model_path))
        model.set_state_dict(state_dict=saved['model_state_dict'])
    if rank not in ('cpu', 'cuda'):
        model = paddle.DataParallel(layers=model, find_unused_parameters=True)
    if print_model == True and rank in (0, 'cpu', 'cuda'):
        model_summary(model)
    return model


def loader_setup(train_ratio, val_ratio, test_ratio, batch_size, dataset,
    rank, seed, world_size=0, num_workers=0):
    train_dataset, val_dataset, test_dataset = process.split_data(dataset,
        train_ratio, val_ratio, test_ratio, seed)
    if rank not in ('cpu', 'cuda'):
        train_sampler = paddle.io.DistributedBatchSampler(dataset=
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            batch_size=1)
    elif rank in ('cpu', 'cuda'):
        train_sampler = None
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle
        =train_sampler is None, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler)
    if rank in (0, 'cpu', 'cuda'):
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True)
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader, test_loader, train_sampler,
        train_dataset, val_dataset, test_dataset)


def loader_setup_CV(index, batch_size, dataset, rank, world_size=0,
    num_workers=0):
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = paddle.io.ConcatDataset(datasets=train_dataset)
    test_dataset = dataset[index]
    if rank not in ('cpu', 'cuda'):
        train_sampler = paddle.io.DistributedBatchSampler(dataset=
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            batch_size=1)
    elif rank in ('cpu', 'cuda'):
        train_sampler = None
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle
        =train_sampler is None, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler)
    if rank in (0, 'cpu', 'cuda'):
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, test_loader, train_sampler, train_dataset,
        test_dataset)


def train_regular(rank, world_size, data_path, job_parameters=None,
    training_parameters=None, model_parameters=None):
    ddp_setup(rank, world_size)
    if rank not in ('cpu', 'cuda'):
        model_parameters['lr'] = model_parameters['lr'] * world_size
    dataset = process.get_dataset(data_path, training_parameters[
        'target_index'], False)
    if rank not in ('cpu', 'cuda'):
        paddle.distributed.barrier()
    (train_loader, val_loader, test_loader, train_sampler, train_dataset, _, _
        ) = (loader_setup(training_parameters['train_ratio'],
        training_parameters['val_ratio'], training_parameters['test_ratio'],
        model_parameters['batch_size'], dataset, rank, job_parameters[
        'seed'], world_size))
    model = model_setup(rank, model_parameters['model'], model_parameters,
        dataset, job_parameters['load_model'], job_parameters['model_path'],
        model_parameters.get('print_model', True))
    optimizer = getattr(paddle.optimizer, model_parameters['Optimizer'])(model.
        parameters(), lr=model_parameters['learning_rate'], **model_parameters[
        'Optimizer_args'])
    scheduler = getattr(paddle.optimizer.lr, model_parameters['LRScheduler']
        )(optimizer, **model_parameters['LRScheduler_args'])
    model = trainer(rank, world_size, model, optimizer, scheduler,
        training_parameters['loss'], train_loader, val_loader,
        train_sampler, model_parameters['epochs'], training_parameters[
        'verbosity'], 'my_model_temp.pth')
    if rank in (0, 'cpu', 'cuda'):
        train_error = val_error = test_error = float('NaN')
        train_loader = DataLoader(train_dataset, batch_size=
            model_parameters['batch_size'], shuffle=False, num_workers=0,
            pin_memory=True)
        train_error, train_out = evaluate(train_loader, model,
            training_parameters['loss'], rank, out=True)
        print('Train Error: {:.5f}'.format(train_error))
        if val_loader != None:
            val_error, val_out = evaluate(val_loader, model,
                training_parameters['loss'], rank, out=True)
            print('Val Error: {:.5f}'.format(val_error))
        if test_loader != None:
            test_error, test_out = evaluate(test_loader, model,
                training_parameters['loss'], rank, out=True)
            print('Test Error: {:.5f}'.format(test_error))
        if job_parameters['save_model'] == 'True':
            if rank not in ('cpu', 'cuda'):
                paddle.save(obj={'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'full_model': model}, path=job_parameters['model_path'])
            else:
                paddle.save(obj={'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'full_model': model}, path=job_parameters['model_path'])
        if job_parameters['write_output'] == 'True':
            write_results(train_out, str(job_parameters['job_name']) +
                '_train_outputs.csv')
            if val_loader != None:
                write_results(val_out, str(job_parameters['job_name']) +
                    '_val_outputs.csv')
            if test_loader != None:
                write_results(test_out, str(job_parameters['job_name']) +
                    '_test_outputs.csv')
        if rank not in ('cpu', 'cuda'):
            paddle.distributed.destroy_process_group()
        error_values = np.array((train_error.cpu(), val_error.cpu(),
            test_error.cpu()))
        if job_parameters.get('write_error') == 'True':
            np.savetxt(job_parameters['job_name'] + '_errorvalues.csv',
                error_values[np.newaxis, ...], delimiter=',')
        return error_values


def predict(dataset, loss, job_parameters=None):
    rank = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
        ).replace('cuda', 'gpu')
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers
        =0, pin_memory=True)
    assert os.path.exists(job_parameters['model_path']
        ), 'Saved model not found'
    if str(rank) == 'cpu':
        saved = paddle.load(path=str(job_parameters['model_path']))
    else:
        saved = paddle.load(path=str(job_parameters['model_path']))
    model = saved['full_model']
    model = model.to(rank)
    model_summary(model)
    time_start = time.time()
    test_error, test_out = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start
    print('Evaluation time (s): {:.5f}'.format(elapsed_time))
    if job_parameters['write_output'] == 'True':
        write_results(test_out, str(job_parameters['job_name']) +
            '_predicted_outputs.csv')
    return test_error


def train_CV(rank, world_size, data_path, job_parameters=None,
    training_parameters=None, model_parameters=None):
    job_parameters['load_model'] = 'False'
    job_parameters['save_model'] = 'False'
    job_parameters['model_path'] = None
    ddp_setup(rank, world_size)
    if rank not in ('cpu', 'cuda'):
        model_parameters['lr'] = model_parameters['lr'] * world_size
    dataset = process.get_dataset(data_path, training_parameters[
        'target_index'], False)
    cv_dataset = process.split_data_CV(dataset, num_folds=job_parameters[
        'cv_folds'], seed=job_parameters['seed'])
    cv_error = 0
    for index in range(0, len(cv_dataset)):
        if index == 0:
            model = model_setup(rank, model_parameters['model'],
                model_parameters, dataset, job_parameters['load_model'],
                job_parameters['model_path'], print_model=True)
        else:
            model = model_setup(rank, model_parameters['model'],
                model_parameters, dataset, job_parameters['load_model'],
                job_parameters['model_path'], print_model=False)
        optimizer = getattr(paddle.optimizer, model_parameters['Optimizer'])(model.
        parameters(), lr=model_parameters['learning_rate'], **model_parameters[
        'Optimizer_args'])
        scheduler = getattr(paddle.optimizer.lr, model_parameters['LRScheduler']
        )(optimizer, **model_parameters['LRScheduler_args'])
        train_loader, test_loader, train_sampler, train_dataset, _ = (
            loader_setup_CV(index, model_parameters['batch_size'],
            cv_dataset, rank, world_size))
        model = trainer(rank, world_size, model, optimizer, scheduler,
            training_parameters['loss'], train_loader, None, train_sampler,
            model_parameters['epochs'], training_parameters['verbosity'],
            'my_model_temp.pth')
        if rank not in ('cpu', 'cuda'):
            paddle.distributed.barrier()
        if rank in (0, 'cpu', 'cuda'):
            train_loader = DataLoader(train_dataset, batch_size=
                model_parameters['batch_size'], shuffle=False, num_workers=
                0, pin_memory=True)
            train_error, train_out = evaluate(train_loader, model,
                training_parameters['loss'], rank, out=True)
            print('Train Error: {:.5f}'.format(train_error))
            test_error, test_out = evaluate(test_loader, model,
                training_parameters['loss'], rank, out=True)
            print('Test Error: {:.5f}'.format(test_error))
            cv_error = cv_error + test_error
            if index == 0:
                total_rows = test_out
            else:
                total_rows = np.vstack((total_rows, test_out))
    if rank in (0, 'cpu', 'cuda'):
        if job_parameters['write_output'] == 'True':
            if test_loader != None:
                write_results(total_rows, str(job_parameters['job_name']) +
                    '_CV_outputs.csv')
        cv_error = cv_error / len(cv_dataset)
        print('CV Error: {:.5f}'.format(cv_error))
    if rank not in ('cpu', 'cuda'):
        paddle.distributed.destroy_process_group()
    return cv_error


def train_repeat(data_path, job_parameters=None, training_parameters=None,
    model_parameters=None):
    world_size = paddle.device.cuda.device_count()
    job_name = job_parameters['job_name']
    model_path = job_parameters['model_path']
    job_parameters['write_error'] = 'True'
    job_parameters['load_model'] = 'False'
    job_parameters['save_model'] = 'False'
    for i in range(0, job_parameters['repeat_trials']):
        job_parameters['seed'] = np.random.randint(1, 1000000.0)
        if i == 0:
            model_parameters['print_model'] = True
        else:
            model_parameters['print_model'] = False
        job_parameters['job_name'] = job_name + str(i)
        job_parameters['model_path'] = str(i) + '_' + model_path
        if world_size == 0:
            print('Running on CPU - this will be slow')
            training.train_regular('cpu', world_size, data_path,
                job_parameters, training_parameters, model_parameters)
        elif world_size > 0:
            if job_parameters['parallel'] == 'True':
                print('Running on', world_size, 'GPUs')
                paddle.distributed.spawn(func=training.train_regular, args=
                    (world_size, data_path, job_parameters,
                    training_parameters, model_parameters), nprocs=
                    world_size, join=True)
            if job_parameters['parallel'] == 'False':
                print('Running on one GPU')
                training.train_regular('cuda', world_size, data_path,
                    job_parameters, training_parameters, model_parameters)
    print('Individual training finished.')
    print('Compiling metrics from individual trials...')
    error_values = np.zeros((job_parameters['repeat_trials'], 3))
    for i in range(0, job_parameters['repeat_trials']):
        filename = job_name + str(i) + '_errorvalues.csv'
        error_values[i] = np.genfromtxt(filename, delimiter=',')
    mean_values = [np.mean(error_values[:, 0]), np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2])]
    std_values = [np.std(error_values[:, 0]), np.std(error_values[:, 1]),
        np.std(error_values[:, 2])]
    print('Training Error Avg: {:.3f}, Training Standard Dev: {:.3f}'.
        format(mean_values[0], std_values[0]))
    print('Val Error Avg: {:.3f}, Val Standard Dev: {:.3f}'.format(
        mean_values[1], std_values[1]))
    print('Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}'.format(
        mean_values[2], std_values[2]))
    if job_parameters['write_output'] == 'True':
        with open(job_name + '_all_errorvalues.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['', 'Training', 'Validation', 'Test'])
            for i in range(0, len(error_values)):
                csvwriter.writerow(['Trial ' + str(i), error_values[i, 0],
                    error_values[i, 1], error_values[i, 2]])
            csvwriter.writerow(['Mean', mean_values[0], mean_values[1],
                mean_values[2]])
            csvwriter.writerow(['Std', std_values[0], std_values[1],
                std_values[2]])
    elif job_parameters['write_output'] == 'False':
        for i in range(0, job_parameters['repeat_trials']):
            filename = job_name + str(i) + '_errorvalues.csv'
            os.remove(filename)


def tune_trainable(config, checkpoint_dir=None, data_path=None):
    from ray import tune
    print('Hyperparameter trial start')
    hyper_args = config['hyper_args']
    job_parameters = config['job_parameters']
    processing_parameters = config['processing_parameters']
    training_parameters = config['training_parameters']
    model_parameters = config['model_parameters']
    model_parameters = {**model_parameters, **hyper_args}
    processing_parameters = {**processing_parameters, **hyper_args}
    world_size = 1
    rank = 'cpu'
    if paddle.device.cuda.device_count() >= 1:
        rank = 'cuda'
    if job_parameters['reprocess'] == 'True':
        time = datetime.now()
        processing_parameters['processed_path'] = time.strftime('%H%M%S%f')
        processing_parameters['verbose'] = 'False'
    data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.
        realpath(__file__))))
    data_path = os.path.join(data_path, processing_parameters['data_path'])
    data_path = os.path.normpath(data_path)
    print('Data path', data_path)
    dataset = process.get_dataset(data_path, training_parameters[
        'target_index'], job_parameters['reprocess'], processing_parameters)
    (train_loader, val_loader, test_loader, train_sampler, train_dataset, _, _
        ) = (loader_setup(training_parameters['train_ratio'],
        training_parameters['val_ratio'], training_parameters['test_ratio'],
        model_parameters['batch_size'], dataset, rank, job_parameters[
        'seed'], world_size))
    model = model_setup(rank, model_parameters['model'], model_parameters,
        dataset, False, None, False)
    optimizer = getattr(paddle.optimizer, model_parameters['Optimizer'])(model.
        parameters(), lr=model_parameters['learning_rate'], **model_parameters[
        'Optimizer_args'])
    scheduler = getattr(paddle.optimizer.lr, model_parameters['LRScheduler']
        )(optimizer, **model_parameters['LRScheduler_args'])
    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = paddle.load(path=
            str(os.path.join(checkpoint_dir, 'checkpoint')))
        model.set_state_dict(state_dict=model_state)
        optimizer.set_state_dict(state_dict=optimizer_state)
        scheduler.set_state_dict(state_dict=scheduler_state)
    for epoch in range(1, model_parameters['epochs'] + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_error = train(model, optimizer, train_loader,
            training_parameters['loss'], rank=rank)
        val_error = evaluate(val_loader, model, training_parameters['loss'],
            rank=rank, out=False)
        if epoch == model_parameters['epochs']:
            if job_parameters['reprocess'] == 'True' and job_parameters[
                'hyper_delete_processed'] == 'True':
                shutil.rmtree(os.path.join(data_path, processing_parameters
                    ['processed_path']))
            print('Finished Training')
        if epoch % job_parameters['hyper_iter'] == 0:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                paddle.save(obj=(model.state_dict(), optimizer.state_dict(),
                    scheduler.state_dict()), path=path)
            tune.report(loss=val_error.cpu().numpy() * 1)


def tune_setup(hyper_args, job_parameters, processing_parameters,
    training_parameters, model_parameters):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune import CLIReporter
    ray.init()
    data_path = '_'
    local_dir = 'ray_results'
    gpus_per_trial = 1
    search_algo = HyperOptSearch(metric='loss', mode='min', n_initial_points=5)
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=
        job_parameters['hyper_concurrency'])
    if os.path.exists(local_dir + '/' + job_parameters['job_name']
        ) and os.path.isdir(local_dir + '/' + job_parameters['job_name']):
        if job_parameters['hyper_resume'] == 'False':
            resume = False
        elif job_parameters['hyper_resume'] == 'True':
            resume = True
    else:
        resume = False
    parameter_columns = [element for element in hyper_args.keys() if 
        element not in 'global']
    parameter_columns = ['hyper_args']
    reporter = CLIReporter(max_progress_rows=20, max_error_rows=5,
        parameter_columns=parameter_columns)
    tune_result = tune.run(partial(tune_trainable, data_path=data_path),
        resources_per_trial={'cpu': 1, 'gpu': gpus_per_trial}, config={
        'hyper_args': hyper_args, 'job_parameters': job_parameters,
        'processing_parameters': processing_parameters,
        'training_parameters': training_parameters, 'model_parameters':
        model_parameters}, num_samples=job_parameters['hyper_trials'],
        search_alg=search_algo, local_dir=local_dir, progress_reporter=
        reporter, verbose=job_parameters['hyper_verbosity'], resume=resume,
        log_to_file=True, name=job_parameters['job_name'], max_failures=4,
        raise_on_failed_trial=False, stop={'training_iteration': 
        model_parameters['epochs'] // job_parameters['hyper_iter']})
    best_trial = tune_result.get_best_trial('loss', 'min', 'all')
    return best_trial


def train_ensemble(data_path, job_parameters=None, training_parameters=None,
    model_parameters=None):
    world_size = paddle.device.cuda.device_count()
    job_name = job_parameters['job_name']
    write_output = job_parameters['write_output']
    model_path = job_parameters['model_path']
    job_parameters['write_error'] = 'True'
    job_parameters['write_output'] = 'True'
    job_parameters['load_model'] = 'False'
    for i in range(0, len(job_parameters['ensemble_list'])):
        job_parameters['job_name'] = job_name + str(i)
        job_parameters['model_path'] = str(i) + '_' + job_parameters[
            'ensemble_list'][i] + '_' + model_path
        if world_size == 0:
            print('Running on CPU - this will be slow')
            training.train_regular('cpu', world_size, data_path,
                job_parameters, training_parameters, model_parameters[
                job_parameters['ensemble_list'][i]])
        elif world_size > 0:
            if job_parameters['parallel'] == 'True':
                print('Running on', world_size, 'GPUs')
                paddle.distributed.spawn(func=training.train_regular, args=
                    (world_size, data_path, job_parameters,
                    training_parameters, model_parameters[job_parameters[
                    'ensemble_list'][i]]), nprocs=world_size, join=True)
            if job_parameters['parallel'] == 'False':
                print('Running on one GPU')
                training.train_regular('cuda', world_size, data_path,
                    job_parameters, training_parameters, model_parameters[
                    job_parameters['ensemble_list'][i]])
    print('Individual training finished.')
    print('Compiling metrics from individual models...')
    error_values = np.zeros((len(job_parameters['ensemble_list']), 3))
    for i in range(0, len(job_parameters['ensemble_list'])):
        filename = job_name + str(i) + '_errorvalues.csv'
        error_values[i] = np.genfromtxt(filename, delimiter=',')
    mean_values = [np.mean(error_values[:, 0]), np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2])]
    std_values = [np.std(error_values[:, 0]), np.std(error_values[:, 1]),
        np.std(error_values[:, 2])]
    for i in range(0, len(job_parameters['ensemble_list'])):
        filename = job_name + str(i) + '_test_outputs.csv'
        test_out = np.genfromtxt(filename, delimiter=',', skip_header=1)
        if i == 0:
            test_total = test_out
        elif i > 0:
            test_total = np.column_stack((test_total, test_out[:, 2]))
    ensemble_test = np.mean(np.array(test_total[:, 2:]).astype(np.float),
        axis=1)
    ensemble_test_error = getattr(F, training_parameters['loss'])(paddle.
        to_tensor(data=ensemble_test), paddle.to_tensor(data=test_total[:, 
        1].astype(np.float)))
    test_total = np.column_stack((test_total, ensemble_test))
    for i in range(0, len(job_parameters['ensemble_list'])):
        print(job_parameters['ensemble_list'][i] + ' Test Error: {:.5f}'.
            format(error_values[i, 2]))
    print('Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}'.format(
        mean_values[2], std_values[2]))
    print('Ensemble Error: {:.5f}'.format(ensemble_test_error))
    if write_output == 'True' or write_output == 'Partial':
        with open(str(job_name) + '_test_ensemble_outputs.csv', 'w') as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_total) + 1):
                if i == 0:
                    csvwriter.writerow(['ids', 'target'] + job_parameters[
                        'ensemble_list'] + ['ensemble'])
                elif i > 0:
                    csvwriter.writerow(test_total[i - 1, :])
    if write_output == 'False' or write_output == 'Partial':
        for i in range(0, len(job_parameters['ensemble_list'])):
            filename = job_name + str(i) + '_errorvalues.csv'
            os.remove(filename)
            filename = job_name + str(i) + '_test_outputs.csv'
            os.remove(filename)


def analysis(dataset, model_path, tsne_args):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    rank = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
        ).replace('cuda', 'gpu')
    inputs = []

    def hook(module, input, output):
        inputs.append(input)
    assert os.path.exists(model_path), 'saved model not found'
    if str(rank) == 'cpu':
        saved = paddle.load(path=str(model_path))
    else:
        saved = paddle.load(path=str(model_path))
    model = saved['full_model']
    model_summary(model)
    print(dataset)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers
        =0, pin_memory=True)
    model.eval()
    model.post_lin_list[0].register_forward_post_hook(hook=hook)
    for data in loader:
        with paddle.no_grad():
            data = data.to(rank)
            output = model(data)
    inputs = [i for sub in inputs for i in sub]
    inputs = paddle.concat(x=inputs)
    inputs = inputs.cpu().numpy()
    print('Number of samples: ', tuple(inputs.shape)[0])
    print('Number of features: ', tuple(inputs.shape)[1])
    targets = dataset.data.y.numpy()
    tsne = TSNE(**tsne_args)
    tsne_out = tsne.fit_transform(inputs)
    rows = zip(dataset.data.structure_id, list(dataset.data.y.numpy()),
        list(tsne_out[:, 0]), list(tsne_out[:, 1]))
    with open('tsne_output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in rows:
            writer.writerow(row)
    fig, ax = plt.subplots()
    main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(main, ax=ax)
    stdev = np.std(targets)
    cbar.mappable.set_clim(np.mean(targets) - 2 * np.std(targets), np.mean(
        targets) + 2 * np.std(targets))
    plt.savefig('tsne_output.png', format='png', dpi=600)
    plt.show()
