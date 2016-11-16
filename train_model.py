import mxnet as mx
import logging
import os

def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()

        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    #if model_prefix is not None:
    #    model_prefix += "-%d" % (kv.rank)
    model_args = {}
    #lr_scale = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
        # TODO: check epoch_size for 'dist_sync'
        epoch_size = args.num_examples / args.batch_size
        model_args['begin_num_update'] = epoch_size * args.load_epoch




    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None



    net_args = network.list_arguments()
    fix_param_names = []
    variable_param_names = []
    for index, name in enumerate(net_args):
        if '__' not in name:
            fix_param_names.append(name)
        else:
            variable_param_names.append(name)


    model = mx.mod.Module(
        symbol = network,
        context = devs,
        fixed_param_names = fix_param_names)



    eval_metrics = ['acc', 'ce']
    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))
    batch_end_callback.append(mx.callback.ProgressBar(args.num_examples * 1.0/ args.batch_size))

    epoch_end_callback = [checkpoint]
    model.fit(train,
            eval_data = val,
            optimizer_params = {"learning_rate":args.lr, "momentum":0.9},
            num_epoch=args.num_epochs,
            arg_params = model_args['arg_params'],
            aux_params = model_args['aux_params'],
            initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
            allow_missing = True,
            batch_end_callback = batch_end_callback,
            epoch_end_callback = epoch_end_callback,
            eval_metric = eval_metrics)







    '''


    model = mx.mod.Module(
        context            = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        allow_extra_params = True)
        #**model_args)
    lr_scale = {}
    if args.finetune is True and args.load_epoch is not None:
        net_args = network.list_arguments()
        for i, name in enumerate(net_args):
            if '__' in name:
                lr_scale[name] = args.finetune_scale
    #        else:
    #            lr_scale[name] = 0
    print lr_scale


    model.fit(
            train,
            eval_data = val,
            num_epoch = args.num_epochs,
            optimizer_params = {'learning_rate':args.lr, 'momentum':0.9},
            batch_end_callback = batch_end_callback,
            epoch_end_callback = checkpoint)


    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
    '''
