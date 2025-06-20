import sys
import argparse
import socket
import enhancedDriver
import json
import time

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost', help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001, help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR', help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1, help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0, help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None, help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3, help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')



try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

sock.settimeout(5.0)  # Increased timeout to 5 seconds for debugging

shutdownClient = False
curEpisode = 0
verbose = True  # Enable verbose mode to see all messages

track_type = arguments.track if arguments.track is not None else 'none'
d = enhancedDriver.Driver(arguments.stage, track_type= track_type, model_type="decision_tree")


# Open a file to store data
data_file = open(f'torcs_data_{int(time.time())}.jsonl', 'w')

while not shutdownClient:
    while True:
        buf = arguments.id + d.init()
        
        try:
            sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf_str = buf.decode('utf-8')
            print('Received:', buf_str)  # Log every received message
        except socket.error as msg:
            continue
        
        if buf_str.find('***identified***') >= 0:
            print('Successfully identified!')
            break

    currentStep = 0
    
    while True:
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf_str = buf.decode('utf-8')
            print('Received in driving loop:', buf_str)  # Log received data
        except socket.error as msg:
            print("Didn't get response from server in driving loop...")
            break  # Exit the loop if no response

        if verbose:
            print('Received:', buf_str)
        
        if buf != None and buf_str.find('***shutdown***') >= 0:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf != None and buf_str.find('***restart***') >= 0:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf != None:
                # Parse the state
                state = d.parser.parse(buf_str)
                action = {
                    'steer': d.control.getSteer(),
                    'accel': d.control.getAccel(),
                    'brake': d.control.getBrake(),
                    'gear': d.control.getGear()
                }
                buf = d.drive(buf_str)
                # Log state-action pair
                data_entry = {'state': state, 'action': action}
                print('Sending:', buf)  # Log sent data
        else:
            buf = '(meta 1)'
        
        if buf != None:
            try:
                sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

data_file.close()
sock.close()