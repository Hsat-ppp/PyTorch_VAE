# methodology
likelihood_x = 'bernoulli'

# configs
parallel_gpu = True

# データ処理や学習に関する定数
num_workers = 16        # num of worker in data loader
lrec = 1.0  # 重み：復元ロス
lkld = 1.0  # 重み：KL divergece (only for VAE)

# AEネットワーク構造に関する定数
i_channel = 1           # 入力チャネル数
o_channel = 64          # 拡大されたチャネル数
width = 28             # 横幅
height = 28            # 縦幅
c_num = 1               # 畳み込み数(各画像次元における)
s_num = 2               # サンプリング(1回につきサイズは半分になる)
fc1_nodes = 128        # fc層1のノード数
dim_latent = 4         # 潜在空間の次元
