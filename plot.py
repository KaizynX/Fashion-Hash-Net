import matplotlib
import matplotlib.pyplot as plt

FOLDER = './fig/'
dataset = [100, 10, 5, '5h']

retrain = {
  'FITB': {},
  'AUC': {},
  'NDCG': {},
}

tradition = {
  'FITB': [0.5581, 0.5427, 0.5561],
  'AUC': {},
  'NDCG': {},
}

'''
def plot_top():
    label = [630, 300, 100, 50, 10]
    xlabel = 'NDCG'
    ylabel = 'AUC'
    x = [0.6564, 0.6489, 0.6427, 0.6444, 0.6348]
    y = [0.8064, 0.8009, 0.8004, 0.7992, 0.7934]

    fig = plt.figure()
    # gs = fig.add_gridspec(1, 1)
    # ax = fig.add_subplot(gs[0, 0])
    margin = 0.01
    plt.xlim(min(x) - margin, max(x) + margin)
    plt.ylim(min(y) - margin, max(y) + margin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, marker='o', label='Top K=')
    bar = plt.bar(x, y, color='white')
    plt.bar_label(bar, labels=label)
    plt.legend()
    plt.plot()
    plt.show()
    fig.savefig(FOLDER + 'top.png')
'''

def plot_top():
  K = [630, 300, 100, 50, 10]
  NDCG = [0.6564, 0.6489, 0.6427, 0.6444, 0.6348]
  AUC = [0.8064, 0.8009, 0.8004, 0.7992, 0.7934]
  metric = ['NDCG', 'AUC']
  k_labels = [f'K={k}' for k in K]

  margin = 0.01
  margin_x = 10
  margin_y = 0.01
  # fig = plt.figure()
  fig = plt.figure(figsize=(16, 8))
  gs = fig.add_gridspec(1, 2)
  for i, data in enumerate((NDCG, AUC)):
    ax = fig.add_subplot(gs[0, i])
    x = K
    y = data
    ax.set_xlim(0, max(x) + 50)
    ax.set_ylim(min(y) - margin_y, max(y) + margin_y)
    # ax.set_title()
    ax.set_xlabel('K')
    ax.set_ylabel(metric[i])
    ax.plot(x, y, marker='o')
    # bar = ax.bar(x, y, color='white')
    bar = ax.bar(x, y)
    ax.bar_label(bar, labels=k_labels)
    # ax.legend()
  plt.plot()
  plt.show()
  fig.savefig(FOLDER + 'top.png')


def plot_hyper():
  beta = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
  epsilon = [0.3, 0.5, 0.6, 0.7, 1.0]
  data = {
    'beta': {
      'AUC':  [0.8053, 0.8057, 0.8053, 0.8064, 0.8054, 0.8047, 0.7981],
      'NDCG': [0.6576, 0.6593, 0.6579, 0.6564, 0.6536, 0.6508, 0.6359],
    },
    'epsilon': {
      'AUC':  [0.8045, 0.8064, 0.8057, 0.8053, 0.8048],
      'NDCG': [0.6526, 0.6564, 0.6560, 0.6583, 0.6567],
    },
  }
  x_labels = ['beta', 'epsilon']
  y_labels = ['AUC', 'NDCG']

  margin_x = 0.1
  margin_y = 0.002
  fig = plt.figure()
  # fig = plt.figure(figsize=(16, 8))
  gs = fig.add_gridspec(2, 2)
  for i, x in enumerate((beta, epsilon)):
    for j, y_label in enumerate(data[x_labels[i]]):
      ax = fig.add_subplot(gs[i, j])
      y = data[x_labels[i]][y_label]
      ax.set_xlim(min(x) - margin_x, max(x) + margin_x)
      ax.set_ylim(min(y) - margin_y, max(y) + margin_y)
      # ax.set_title()
      ax.set_xlabel(x_labels[i])
      ax.set_ylabel(y_label)
      ax.plot(x, y, marker='o')
      # bar = ax.bar(x, y, color='white')
      # bar = ax.bar(x, y)
      # ax.bar_label(bar, labels=k_labels)
      # ax.legend()
  plt.plot()
  plt.show()
  fig.savefig(FOLDER + 'hyper.png')


def plot_dataset():
  # x = [5, 5, 10, 100]
  x = [2, 5, 10, 20]
  data = {
    'FITB': {
      'retrain-epoch=1': [0.5993, 0.5967, 0.6157, 0.6199, ],
      'retrain-best': [0.6039, 0.6127, 0.6235, 0.6399, ],
      'similarity': [0.5561, 0.5427, 0.5581, 0.5489, ],
    },
    'AUC': {
      'retrain-epoch=1': [0.8680, 0.8702, 0.8912, 0.9097, ],
      'retrain-best': [0.8787, 0.8817, 0.9068, 0.9225, ],
      'similarity': [0.7932, 0.7953, 0.8032, 0.8048, ],
    },
    'NDCG': {
      'retrain-epoch=1': [0.7726, 0.7763, 0.8025, 0.8345, ],
      'retrain-best': [0.7903, 0.7949, 0.8314, 0.8645, ],
      'similarity': [0.6391, 0.6389, 0.6568, 0.6615, ],
    },
  }

  fig = plt.figure()
  # fig = plt.figure(figsize=(16, 8))
  gs = fig.add_gridspec(1, 3)
  for i, y_label in enumerate(data):
    ax = fig.add_subplot(gs[0, i])
    yl = 1.0
    yr = 0.0
    for label in data[y_label]:
      y = data[y_label][label]
      yl = min(yl, min(y))
      yr = max(yr, max(y))
      ax.plot(x, y, marker='o', label=label)
    # ax.set_xlim(0, max(x) + 50)
    # ax.set_ylim(yl - margin_y, yr + margin_y)
    # ax.set_ylim(yl - 0.1 , yr + 0.005)
    # ax.set_ylim(yl - 0.2 , yr + 0.05)
    ax.set_ylim(yl - 0.05 , yr + 0.01)
    # ax.set_ylim(0 , 1)
    # ax.set_ylim(0 , 1)
    # ax.set_title()
    ax.set_xlabel('dataset size')
    ax.set_ylabel(y_label)
    # bar = ax.bar(x, y, color='white')
    # bar = ax.bar(x, y)
    # ax.bar_label(bar, labels=k_labels)
    ax.legend()
    ax.set_xticks([0, 2, 5, 10, 15, 20])
    ax.set_xticklabels(['0%', '5%*', '5%', '10%', '...', '100%'])

  plt.plot()
  plt.show()
  fig.savefig(FOLDER + 'dataset.png')

def plot_time():
  x = [5, 10, 100]
  y = [17, 23, 88]
  fig = plt.figure()
  # fig = plt.figure(figsize=(16, 8))
  gs = fig.add_gridspec(1, 1)
  ax = fig.add_subplot(gs[0, 0])
  retrain = ax.bar(x, y, label='retrain-GPU')
  tradition = ax.bar(x, [6] * 3, label='tradition-CPU')
  ax.set_xlim(0, 110)
  ax.set_ylim(0, 100)
  # ax.set_ylim(0 , 1)
  # ax.set_title()
  ax.set_xlabel('dataset size')
  ax.set_ylabel('runtime(s)/epoch')
  # bar = ax.bar(x, y, color='white')
  # bar = ax.bar(x, y)
  ax.bar_label(retrain)
  ax.bar_label(tradition)
  ax.legend()
  # ax.set_xticks([0, 5, 10, 20])
  # ax.set_xticklabels(['0%', '5%*', '5%', '10%', '...', '100%'])

  plt.plot()
  plt.show()
  fig.savefig(FOLDER + 'time.png')


if __name__ == '__main__':
    # plot_top()
    # plot_hyper()
    plot_dataset()
    # plot_time()