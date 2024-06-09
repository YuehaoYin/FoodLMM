import torch
from functools import partial

from utils.dataset import collate_fn


def get_test_loader(args, test_dataset_list, tokenizer):
    test_dataset_list_new = []
    for test_data in test_dataset_list:
        print('test_data', test_data)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data['data'], shuffle=False, drop_last=False
            )
        else:
            val_sampler = torch.utils.data.SequentialSampler(test_data['data'])

        bs = args.val_batch_size
        val_loader = torch.utils.data.DataLoader(
            test_data['data'],
            batch_size=bs,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=partial(
                # collate_fn_val if test_data['type'] == 'seg_ood' else collate_fn,
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=True,
                local_rank=args.local_rank,
            )
        )
        test_data['data'] = val_loader
        test_dataset_list_new.append(test_data)
    return test_dataset_list_new
