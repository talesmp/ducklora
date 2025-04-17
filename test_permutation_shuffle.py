import numpy as np

num_records = 9                                 
indices = np.arange(num_records) #len(dset)
batch_size = 4

for i in range(0, num_records - batch_size + 1, batch_size):

    print("Iteration", i)
    batch_indices = indices[i:i + batch_size]
    print("Batch[",i,"] Original:", batch_indices)       # [0, 1, 2, ...]
    batch = list(batch_indices)
    print("Batch[",i,"] Permutation:", batch)
    print("Batch[",i,"] Length:", len(batch))

    # if True:
    #     new_indices = np.random.permutation(indices)
    #     print("Iter[",i,"] Original:", indices)       # [0, 1, 2, ...]
    #     print("Iter[",i,"] Permutation:", new_indices)  

    # print()
    # print(len(indices))
    # print(len(new_indices))

    # if True:
    #     np.random.shuffle(indices)  
    #     print("Iter[",i,"] Shuffled:", indices)


# for i in range(0, num_records - batch_size + 1, batch_size):
#     batch_indices = indices[i:i + batch_size]
#     print("Batch[",i,"] Original:", batch_indices)       # [0, 1, 2, ...]
#     batch = list(batch_indices)
#     lengths = [len(x) for x in batch]
#     batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

#     for j in range(batch_size):
#         print("Batch[",i,"] Permutation:", batch[j])
#         print("Batch[",i,"] Length:", lengths[j])
#         batch_arr[j, :lengths[j]] = batch[j]