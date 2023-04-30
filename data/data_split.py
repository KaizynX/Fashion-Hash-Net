'''
Author: Kaizyn
Date: 2023-04-17 22:09:54
LastEditTime: 2023-04-30 17:06:00
'''
import os

FILE_PATH = os.path.abspath(__file__)
DATA_FOLDER = os.path.join(os.path.dirname(FILE_PATH), 'polyvore')


def main():
  dataset = input('Input data set: ')
  cutoff = input('Input cutoff %: ')
  limit = input('Input hard limit: ')
  dataset = dataset or 'tuples_53'
  cutoff = int(cutoff) if cutoff else 10
  limit = int(limit) if limit else 1000000
  dataset_folder = os.path.join(DATA_FOLDER, dataset)
  new_folder = os.path.join(DATA_FOLDER, f'{dataset}_{cutoff}')
  os.system(f'mkdir {new_folder}')
  for file_name in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, file_name)
    new_path = os.path.join(new_folder, file_name)
    if file_name[:12] != 'tuples_train':
      os.system(f'cp {file_path} {new_path}')
    else:
      cnt = {}
      cut = {}
      with open(file_path, 'r') as f:
        data = f.readlines()
      for line in data[1:]:
        user = line.split(',', maxsplit=1)[0]
        cnt[user] = cnt.get(user, 0) + 1
      print(f'-----split {file_path}-----')
      tot = 0
      for key, val in cnt.items():
        # print(f'user {key} has {val} records.')
        cut[key] = min(val * cutoff // 100, limit)
        tot += val
      print(f'average {tot / len(cnt)} records/user')

      with open(new_path, 'w') as f:
        f.write(data[0])
        for line in data[1:]:
          user = line.split(',', maxsplit=1)[0]
          if cut[user] > 0:
            f.write(line)
            cut[user] -= 1
      

if __name__ == '__main__':
  main()