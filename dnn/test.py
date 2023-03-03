from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():

    dataset = get_dataset()
    global_step = 0
    
    m = nn.DataParallel(Model())

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    #pos_weight = t.FloatTensor([5.]).cuda()
    #writer = SummaryWriter()
    
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8, collate_fn=collate_fn_transformer)
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        pbar.set_description("Processing at epoch")
        global_step += 1
        if global_step < 400000:
            adjust_learning_rate(optimizer, global_step)
                
        source_mel, target_mel, source_mel_input, target_mel_input, source_pos_mel, target_pos_mel = data
        #stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
        mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(source_mel_input, source_pos_mel, target_mel_input, target_pos_mel)
        
        mel_loss = nn.L1Loss()(mel_pred, target_mel)
        post_mel_loss = nn.L1Loss()(postnet_pred, target_mel)
        loss = mel_loss + post_mel_loss

        optimizer.zero_grad()
        # Calculate gradients
        loss.backward()

        nn.utils.clip_grad_norm_(m.parameters(), 1.)

        # Update weights
        optimizer.step()
        print("finish")

    


if __name__ == '__main__':
    main()
