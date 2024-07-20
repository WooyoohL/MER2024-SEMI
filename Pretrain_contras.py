import torch
import torch.nn as nn
from toolkit.data.noise_scheduler import NoiseScheduler
import torch.nn.functional as F


class Pretrain_contras(nn.Module):
    def __init__(self, args):
        super(Pretrain_contras, self).__init__()
        self.batch_size = args.batch_size
        self.audio_dim = args.audio_dim
        self.text_dim = args.text_dim
        self.video_dim = args.video_dim
        self.output_dim1 = args.output_dim1
        self.output_dim2 = args.output_dim2
        self.dropout = args.dropout  # 0
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.dropout = args.dropout
        self.attn_mask = True
        self.cls_layers = '256, 128'
        self.grad_clip = args.grad_clip
        self.timestep = args.num_timestep
        self.temperature1 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature2 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature3 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature4 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature5 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature6 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature7 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature8 = torch.nn.Parameter(torch.tensor(1.))
        self.temperature9 = torch.nn.Parameter(torch.tensor(1.))
        self.isTrain = True
        timestep_set = [[180, 40, 140], [200, 200, 200], [100, 100, 100], [25, 25, 25]]
        timestep = timestep_set[2]
        print('Noise timestep: ', timestep)
        self.noise_scheduler_A = NoiseScheduler(noise_type=self.noise_type, num_time_steps=timestep[0],
                                                beta_start=0.001, beta_end=0.1)
        self.noise_scheduler_V = NoiseScheduler(noise_type=self.noise_type, num_time_steps=timestep[1],
                                                beta_start=0.001, beta_end=0.1)
        self.noise_scheduler_L = NoiseScheduler(noise_type=self.noise_type, num_time_steps=timestep[2],
                                                beta_start=0.001, beta_end=0.1)

        self.netA = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.audio_dim,
                                           nhead=self.num_heads,
                                           dim_feedforward=2048,
                                           dropout=self.dropout),
                                          num_layers=self.layers)
        self.netV = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.video_dim,
                                           nhead=self.num_heads,
                                           dim_feedforward=2048,
                                           dropout=self.dropout),
                                          num_layers=self.layers)
        self.netT = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.text_dim,
                                           nhead=self.num_heads,
                                           dim_feedforward=2048,
                                           dropout=self.dropout),
                                          num_layers=self.layers)

        self.linearA = nn.Linear(self.audio_dim, self.hidden_dim)
        self.linearV = nn.Linear(self.video_dim, self.hidden_dim)
        self.linearT = nn.Linear(self.text_dim, self.hidden_dim)
        self.linearA1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.linearV1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.linearT1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

        self.netInv = torch.nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.LeakyReLU())

    def forward(self, batch):
        x_l = batch['texts']
        x_a = batch['audios']
        x_v = batch['videos']

        feat_A = self.netA(x_a.unsqueeze(1))
        feat_V = self.netV(x_v.unsqueeze(1))
        feat_L = self.netL(x_l.unsqueeze(1))

        feat_A = self.linearA(feat_A.squeeze(1))
        feat_V = self.linearV(feat_V.squeeze(1))
        feat_L = self.linearT(feat_L.squeeze(1))
        if self.training:
            a_arg = self.noise_scheduler_A.add_noise(x_a.unsqueeze(1))
            v_arg = self.noise_scheduler_V.add_noise(x_v.unsqueeze(1))
            l_arg = self.noise_scheduler_L.add_noise(x_l.unsqueeze(1))
            feat_A_arg = self.netA(a_arg)
            feat_V_arg = self.netV(v_arg)
            feat_L_arg = self.netT(l_arg)
            feat_A_arg = self.linearA(feat_A_arg.squeeze(1))
            feat_V_arg = self.linearV(feat_V_arg.squeeze(1))
            feat_L_arg = self.linearT(feat_L_arg.squeeze(1))

        feat_A_concat = self.linearA1(feat_A)
        feat_V_concat = self.linearV1(feat_V)
        feat_L_concat = self.linearT1(feat_L)

        feat_A_inv = self.netInv(feat_A)
        feat_V_inv = self.netInv(feat_V)
        feat_L_inv = self.netInv(feat_L)

        if self.training:
            interloss = self.cal_loss(feat_A, feat_V, feat_L,
                                      feat_A_inv, feat_V_inv, feat_L_inv,
                                      feat_A_arg, feat_V_arg, feat_L_arg,
                                      feat_A_concat, feat_V_concat, feat_L_concat)
        else:
            interloss = torch.tensor(0).cuda()

        return 0, 0, 0, interloss

    def cal_loss(self, feat_A, feat_V, feat_L,
                 feat_A_inv, feat_V_inv, feat_L_inv,
                 feat_A_arg, feat_V_arg, feat_L_arg,
                 feat_A_concat, feat_V_concat, feat_L_concat):
        modality_contrastive_AV = self.nt_xent_loss(feat_A, feat_V, self.temperature1)
        modality_contrastive_AL = self.nt_xent_loss(feat_A, feat_L, self.temperature2)
        modality_contrastive_VL = self.nt_xent_loss(feat_V, feat_L, self.temperature3)

        modality_contrastive_A_VL = self.nt_xent_loss(feat_A_concat, torch.cat([feat_V, feat_L], dim=-1),
                                                      temperature=self.temperature4)
        modality_contrastive_V_AL = self.nt_xent_loss(feat_V_concat, torch.cat([feat_A, feat_L], dim=-1),
                                                      temperature=self.temperature5)
        modality_contrastive_L_AV = self.nt_xent_loss(feat_L_concat, torch.cat([feat_A, feat_V], dim=-1),
                                                      temperature=self.temperature6)

        modality_contrastive_loss = (modality_contrastive_AV + modality_contrastive_VL
                                     + modality_contrastive_AL + modality_contrastive_A_VL
                                     + modality_contrastive_V_AL + modality_contrastive_L_AV) / 6

        noise_contrastive_loss_A = self.nt_xent_loss(feat_A, feat_A_arg, self.temperature7)
        noise_contrastive_loss_V = self.nt_xent_loss(feat_V, feat_V_arg, self.temperature8)
        noise_contrastive_loss_L = self.nt_xent_loss(feat_L, feat_L_arg, self.temperature9)
        noise_contrastive_loss = (noise_contrastive_loss_A + noise_contrastive_loss_V + noise_contrastive_loss_L) / 3

        return modality_contrastive_loss + noise_contrastive_loss 

    def nt_xent_loss(self, z_i, z_j, temperature):
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.T) / temperature
        sim_matrix /= temperature
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -log_probs[mask]
        return loss.mean()
