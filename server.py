#wrote the final code of server here only. Any changes there , do in any other file.

import socket
import time
import numpy as np
import ast

global_weights = np.random.rand(1,7)
global_weights=global_weights[0]

def string_to_nested_list(data_string):

  try:
    # Attempt to evaluate the string as a list using ast.literal_eval
    data_list = ast.literal_eval(data_string)
  except (SyntaxError, ValueError):
    raise ValueError("Invalid list format in string.")

  # Check if the data is a valid list structure (nested lists)
  if not isinstance(data_list, (list, tuple)):
    raise ValueError("Invalid list format in string.")

  return data_list
    

def updating_weights(list1, list2):
  if len(list1) != len(list2):
    raise ValueError("Lists must be the same size.")

  # Handle 1D lists directly
  if isinstance(list1, (list, tuple)) and len(list1) == 1:
    return [(x + y) / 2 for x, y in zip(list1[0], list2[0])]

  # Recursively handle nested lists
  def average_nested(nested_list1, nested_list2):
    if isinstance(nested_list1, (list, tuple)):
      return [average_nested(sub_list1, sub_list2) for sub_list1, sub_list2 in zip(nested_list1, nested_list2)]
    else:
      return (nested_list1 + nested_list2) / 2

  return average_nested(list1, list2)





buffersize = 1024
ServerPort = 2222
ServerIp='192.168.193.103'



#initialize the socket communication
RPIsocket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RPIsocket.bind((ServerIp, ServerPort))
print('Server is Up for Listening')



time.sleep(1)
print("line 60 executed")
client_weights, client_address=RPIsocket.recvfrom(buffersize)
client_weights=client_weights.decode('utf-8')
client_weights=string_to_nested_list(client_weights) 
print(client_weights)
global_Weights=client_weights
global_weights_to_send=str(global_weights)
bytesToSend=global_weights_to_send.encode('utf-8')
print('Client Address', client_address[0])
time.sleep(0.5)
RPIsocket.sendto(bytesToSend, client_address)


#main code
while True:
  time.sleep(1)
  client_weights, client_address=RPIsocket.recvfrom(buffersize)
  client_weights=client_weights.decode('utf-8')
  client_weights=string_to_nested_list(client_weights) 
  print(client_weights)
  global_weights_to_send=updating_weights(client_weights, list2=global_weights)
  print(global_weights)
  global_weights=str(global_weights_to_send)
  bytesToSend=global_weights.encode('utf-8')
  print('Client Address', client_address[0])
  time.sleep(0.5)
  RPIsocket.sendto(bytesToSend, client_address)

  #todo-list
  #re-write the string to nested list function
  # re-write update function
