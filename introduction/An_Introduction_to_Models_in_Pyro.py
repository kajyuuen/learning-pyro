import torch
import pyro

pyro.set_rng_seed(101)

# Primitive Stochastic Functions

# XXX: Primitive Stochastic Functionsの意味がわからない
#       -> いわゆる確率分布ですかね？
# 正規分布N(0, 1)
loc = 0
scale = 1
normal = torch.distributions.Normal(loc, scale)
x = normal.rsample()
print("sample", x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)

## A Simple Model
# ベルヌーイと正規分布からなる混合モデル
def weather():
    # 30% cloudy
    # XXX: sample(), rsample()の違いは？
    cloudy = torch.distributions.Bernoulli(0.3).sample()
    cloudy = "cloudy" if cloudy.item() == 1 else "sunny"
    mean_temp = {"cloudy": 55.0, "sunny": 75.0}[cloudy]
    scale_temp = {"cloudy": 10.0, "sunny": 15.0}[cloudy]
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
    return cloudy, temp.item()

# The pyro.sample Primitive
x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
print(x)
# NOTE: pyro.sampleを使おうねという話
# -> 一つの変数に名前を持ったランダムな変数を複数個持たせられるので

def weather():
    cloudy = pyro.sample("cloudy", pyro.distributions.Bernoulli(0.3))
    cloudy = "cloudy" if cloudy.item() == 1 else "sunny"
    mean_temp = {"cloudy": 55.0, "sunny": 75.0}[cloudy]
    scale_temp = {"cloudy": 10.0, "sunny": 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather())

# Universality: Stochastic Recursion, Higher-order Stochastic Functions, and Random Control Flow


