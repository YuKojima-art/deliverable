# AttentionおよびGRU搭載型Seq2Seqの構築
# 概要
# Seq2SeqにAttention層を加え、さらに通常のRNNやLSTMではなくGRUを搭載したものを構築

import sys
sys.path.append('../kojimayu')
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import os.path

# データをIDに、IDをデータにするものの枠を作る。
id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars): # 新しいデータであれば更新する。
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name='date.txt', seed=1984):
    file_path = '../kojimayu/date/' + file_name # 自身のローカルで行うため、パスは自身のもので構築している。

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, 'r'):
        idx = line.find('_')
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)

    #  x(入力データ)とt(正解ラベル)の配列を作る。
    x = np.zeros((len(questions), len(questions[0])), dtype=np.int)
    t = np.zeros((len(questions), len(answers[0])), dtype=np.int)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # 乱数によりデータをシャッフルする。
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 8：2の割合で訓練データとテストデータを分ける。
    split_at = len(x) - len(x) // 5
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char

# Adam
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

# Trainer
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters+1) * batch_size]
                batch_t = t[iters * batch_size:(iters+1) * batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad) # 勾配爆発への対策メソッド 勾配クリッピングという
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()
        
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
            
    rate = max_norm / (total_norm + 1e-6) 
    if rate < 1:
        for grad in grads:
            grad *= rate
                
def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list ※中身は複製されない※
        
    while True:
        find_flg = False
        L = len(params)
            
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]: # params が同じものを探している(同じである場合の処理は以下)
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j) # pop() は削除を意味する
                    grads.pop(j) # pop() は削除を意味する
                        
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    # np.all() は()内の条件を満たすか確認している
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j) # pop() は削除を意味する
                    grads.pop(j) # pop() は削除を意味する
                        
                if find_flg: break
            if find_flg: break
        if not find_flg: break
                
    return params, grads

# couting program
def eval_seq2seq(model, question, correct, id_to_char, verbose=False, is_reverse=False):
    correct = correct.flatten()
    # 頭の区切り文字
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 文字列への変換
    question = ''.join([id_to_char[int(c)] for c in question.flatten()]) # 質問文
    correct = ''.join([id_to_char[int(c)] for c in correct]) # 実際の解答
    guess = ''.join([id_to_char[int(c)] for c in guess]) # 解答文(文章生成文)

    if verbose:
        if is_reverse:
            question = question[::-1] # question の並びを逆にする

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'} # 色の指定
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess: # 実際の解答と生成した解答が合ってた場合
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = '0'
            print(mark + ' ' + guess)
        else: # その他の場合(不正解の場合)
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0 # 正解していたら1を返し、不正解なら0を返す

# build layers
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        # keepdims=True とは元の配列に対してブロードキャストを正しく適用させ演算する
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3: # 教師ラベルが one_hot_vector の場合 正解のインデックスに変換
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label) # ts が-1でない時(self.ignore_label の数値でない時)

        # バッチ分と時系列分を乗算でまとめる
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label に該当するものは損失を 0 にする (mask の対象外で計算されない)
        loss = -np.sum(ls) # 各時刻ごとの損失 ls を加算していく
        loss /= mask.sum() # 損失の平均値を出す

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        # 該当する dx のインデックスにある要素全てに対して -1 する
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        # [:, np.newaxis] により次元を1つ追加
        #  mask により ignore_label に該当するものは損失を 0 にする (mask の対象外で計算されない)

        dx = dx.reshape((N, T, V)) # 入力データ(N, T, D)と同様の形に整形

        return dx


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1) # x の配列を N*T 行　にする (-1 は元の形状から推測して作られるという指定,今回なら列の長さが対象)
        out = np.dot(rx, W) + b # 全てをまとめて計算し,出力層へ
        self.x = x
        return out.reshape(N, T, -1) # -1 は元の形状から推測して作られるという指定, 今回なら3次元目が対象

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0) # 列方向に(axis=0) dout を加算
        dW = np.dot(rx.T, dout) # 転置(.T)して計算
        dx = np.dot(dout, W.T) # 転置(.T)して計算
        dx = dx.reshape(* x.shape) # x の形状に変換

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)] # W と同じ形状の配列で全ての要素0にする
        self.idx = None
        # idx は重み(単語の分散表現)を取り出すときに使う数値
        # idx には単語idを配列として格納していく

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # dW の形状を維持したまま、要素を0にする

        np.add.at(dW, self.idx, dout)
        # np.add.at(A, idx, B) → B を A に加算し、加算を行う A の行を idx によって指定する
        # つまり指定した重み(単語の分散表現)に対してだけ加算を行う (重複した idx があっても加算により正しく処理)
        # optimizer を使用するなら dW[self.idx] = dout
        return None


    
    
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t]) # 各時刻ごとの重みを取り出し,該当する out に入れる
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0] # grad を加算していく

        self.grads[0][...] = grad # 最終的な勾配 grad を上書き

        return None

# Attention and GRU
class GRU:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        '''
        それぞれのデータの形状は以下の通り(計算上最後に求める形状に問題がない事が分かる)
        各パラメータは3つ分を集約
        x = N * D , Wx = D * 3H
        h_prev = N * H , Wh = H * 3H , b = 1 * 3H
        '''
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        # Wx, Wh のパラメータを3つに分ける(z, r, h)
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        # GRU 内部の要素である z, r, h_hat を公式で算出
        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r * h_prev, Whh) + bh)

        # z, r, h_hat を使い h_next を公式で算出(GRU の最後の計算)
        h_next = (1 - z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        # forward と同様にパラメータを3つに分ける(z, r, h) <ただしバイアスはいらない>
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        # forward で行った計算の逆伝播
        dh_hat = dh_next * z
        dh_prev = dh_next * (1 - z) # ①

        # dh_hat の逆伝播 <np.tanh() の逆伝播>
        dt = dh_hat * (1 - h_hat ** 2)
        dbh = np.sum(dt, axis=0) # bh の勾配 : (H,) の形状に戻すために axis=0 方向に合計
        dWhh = np.dot((r * h_prev).T, dt) # Wh の勾配 : 転置を行い乗算
        dhr = np.dot(dt, Whh.T) # (r * h_prev) の勾配 : 転置を行い乗算
        dWxh = np.dot(x.T, dt) # Wxh の勾配 : 転置を行い乗算
        dx = np.dot(dt, Wxh.T) # x の勾配 : 転置を行い乗算
        dh_prev += r * dhr # ①で出した dh_prev に加算

        # z の backward
        dz = dh_next * h_hat - dh_next * h_prev # h_next の算出計算の逆伝播
        dt = dz * z * (1 - z) # sigmoid の逆伝播公式
        dbz = np.sum(dt, axis=0)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T) # ①で出した dh_prev に加算
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T) # dx をまとめるため加算

        # r の backward
        dr = dhr * h_prev
        dt = dr * r * (1 - r) # sigmoid の逆伝播
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T) # dx をまとめるため加算

        # 引数に与えた配列を横方向に連結 (縦方向に連結する時 np.vstack())
        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        # grads を上書き
        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev

class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T): # T回繰り返す(時系列データ分)
            layer = GRU(* self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None
        
        
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

class WeightSum:
    def __init__(self):
        self.params, self.grads = [], [] # このレイヤは学習するパラメータを持たないため [] とする
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        # a は各単語の重要度を表す重み
        ar = a.reshape(N, T, 1).repeat(H, axis=2)# a の形状を hs と同じ(N, T, H)へ変換
        t = hs * ar # 乗算により各単語の重要度を表現したコンテキストベクトルが誕生
        c = np.sum(t, axis=1) # 時系列の形状(T)を消す (N, H)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1) # dt の形状に直す(N, T, H) <sum の逆伝播>
        dhs = dt * ar
        dar = dt * hs
        da = np.sum(dar, axis=2) # da の形状に直す(N, T) <repeat の逆伝播>
        return dhs, da

class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1) # hs と同じ形状の (N, T, H) へ変換
        t = hs * hr # 乗算によりhs と h の対応関係を表現(h が hs にどれだけ似ているかを表現)
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s) # softmax により確率分布で表現

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2) # ds の形状に直す (N, T, H) <sum の逆伝播>
        dhr = dt * hs
        dhs = dt * hr
        dh = np.sum(dhr, axis=1) # dh の形状に直す (N, H) <repeat の逆伝播>

        return dhs, dh

class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a # 各単語への重みを保持
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T): # 時系列分繰り返す
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :]) # 各時系列に該当するデータを入れていく
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
            # Attention で保持した各単語への重みをリスト追加していく → 各時系列ごとの単語への重みを保有する

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t] # t番目(各時系列の順番)のレイヤを呼び出す
            dhs, dh = layer.backward(dout[:, t, :]) # 各時系列に該当する dout をいれる
            dhs_enc += dhs # 各時系列ごとの dhs をまとめる(encoder の hs の勾配として)
            dhs_dec[:, t, :] = dh # 各時系列ごとの dh をまとめる(decoder の h の勾配として)

        return dhs_enc, dhs_dec

# Seq2Seq
class AttentionEncoderGRU:
    '''
    vocab_size 語彙数(文字の種類)
    wordvec_size 文字ベクトルの次元数
    hidden_size 隠れ状態ベクトルの次元数
    '''
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D ,H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        gru_Wx = (rn(D, 3 * H) / np.sqrt(D)).astype('f')
        gru_Wh = (rn(H, 3 * H) / np.sqrt(H)).astype('f')
        gru_b = np.zeros(3 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=False)
        # stateful=False の理由：今回は短い時系列データが複数存在し、隠れ状態ベクトルを維持しないため

        self.params = self.embed.params + self.gru.params
        self.grads = self.embed.grads + self.gru.grads
        self.hs = None
    
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.gru.forward(xs)
        return hs # hs の時系列データとして全て Decoder に渡す
        
    def backward(self, dhs):
        dout = self.gru.backward(dhs) # dh は Decoder より渡された勾配
        dout = self.embed.backward(dout)
        return dout
    
class AttentionDecoderGRU:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        gru_Wx = (rn(D, 3 * H) / np.sqrt(D)).astype('f')
        gru_Wh = (rn(H, 3 * H) / np.sqrt(H)).astype('f')
        gru_b = np.zeros(3 * H).astype('f')
        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.gru = TimeGRU(gru_Wx, gru_Wh, gru_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.gru, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1] # 一番最後の enc_hs を取り出す
        self.gru.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.gru.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2) # self.gru で算出した dec_hs と c を連結
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H  = H2 // 2

        # ①self.gru からもらった hs, attention で算出した c に分ける
        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        # ②attention に入っていった encoder の hs と decoder(self.gruで算出された) の hs に分ける
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        # ①②で出した self.gru から出てきた ds0,1 を合流させる
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.gru.backward(ddec_hs)
        dh = self.gru.dh
        denc_hs[:, -1] += dh # self.gru.dh は encoder の最後の hs であり, これも encoder へ渡す 
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.gru.set_state(h) # enc_hs の最後の h をセットする

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1)) # 生成物(sample_id)を入力データとして使うためのコード
            out = self.embed.forward(x)
            dec_hs = self.gru.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten()) # argmax 今回は確率的でなく決定して出力するため
            sampled.append(sample_id)

        return sampled # 生成した単語のidリスト

# Attention Seq2Seq
class AttentionSeq2seqGRU:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoderGRU(*args)
        self.decoder = AttentionDecoderGRU(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh) # Encoder に dh を渡す
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sample = self.decoder.generate(h, start_id, sample_size)
        return sample

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]

# Trainer fit
# データの取り出し
(x_train, t_train), (x_test, t_test) = load_data('date.txt')
char_to_id, id_to_char = get_vocab()

# データを反転させる(学習精度の向上)
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 64
max_epoch = 4
max_grad = 5.0

model = AttentionSeq2seqGRU(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

model.save_params()

# グラフの描画(学習内容の評価)
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()
