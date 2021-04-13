import torch
import numpy as np
import time
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--w", help="word size", default=4, type=int)
  parser.add_argument("--file", help="uncorrected fasta file", default='10k_pbsim.fasta', type=str)  
  return parser.parse_args()

def encode(text, k, char2int):

  encoded = np.array([char2int[text[i:i+k]] for i in range(0,len(text)-k+1)])
  encoded = torch.from_numpy(encoded).cuda()
  
  return encoded

def main(args):
  text = "ATCGATCG"
  k = args.w
  n_vocab = 4**k
  key0 = "A"
  key1 = "a"
  for i in range(k-1):
      key0 = key0+"A"
      key1 = key1+"a"
  
  char2int = {key0:0,key1:0}
  
  bits4 = {'00':'0', '01':'1', '10':'2', '11':'3'}
  bits2charU = {'00':'A', '01':'T', '10':'C', '11':'G'}
  bits2charL = {'00':'a', '01':'t', '10':'c', '11':'g'}
  n_bits = 2*k
  for i in range (1,n_vocab):
      b = bin(i)[2:]
      n_zeroes = 2*k-len(b)
      z = ''
      for j in range (n_zeroes):
          z = z+'0'
      b = z+b
      
      B4 = ''
      b4 = ''
      for j in range(0,2*k,2):
          B4 = B4+bits2charU[b[j:j+2]]
          b4 = b4+bits2charL[b[j:j+2]]
      char2int[B4] = i
      char2int[b4] = i
  
  
  start_time = time.time()
  with open (args.file) as f:
      reads = f.read().splitlines()
  
  bptt = 99
  window = bptt+1
  stride = 50
  batch_size = 64
  
  data = torch.empty(bptt,0).cuda()
  targets = torch.tensor([]).cuda()
  print ("Number of reads present", len(reads)/2)
  for i in range (1,len(reads),2):
      if (i%100 == 1): print("Number of reads processed: ",int(i/2)) 
      l = len(reads[i])
      for j in range(0, stride*int((l-window-k+1)/stride), stride):
          sequence = encode(reads[i][j:j+window+k-1], k, char2int)
          data = torch.cat((data, sequence[0:bptt].view(bptt,-1).type(torch.float)), dim=1)
          targets = torch.cat((targets, sequence[1:1+bptt].type(torch.float)), dim=0)
          
  
        
  torch.save(data, "train_data.pt")
  torch.save(targets, "train_targets.pt")
  print ("Time taken = ", time.time() - start_time)

if __name__ == "__main__":
  args = parse_args()
  main(args)
