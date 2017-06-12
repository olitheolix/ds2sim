# Installation
## Without Server
```
>> pip install ds2client
>> generate_dataset  # Calls anaconda/envs/ds2client/bin/generate_dataset.py
```
Then run deep learning on the generated dataset.

## With Server
To download and install the dependencies, run

```
>> git clone https://github.com./olitheolix/ds2server
>> cd ds2server/dependencies
>> ./download_dependencies.sh
>> ./compile_and_install.sh
```
NOTE: this does _not_ install anything into system folders (everything is
installed into the repository folder), and you do not need to to be root for
this operation.

Then compile the Python wrappers.
```
>> cd ds2server/ds2server/horde
>> make
```

Move to the main directory and run the unit tests.
```
>> cd ds2server/
>> py.test
```


### Usage
Start the server in its own terminal.

```
python server.py
```

In a new terminal, start the viewer:
```
>> python viewer.py
```
