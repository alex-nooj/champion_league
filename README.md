# Champion League
## Setup
### 1. Pokemon Showdown
Pokemon Showdown is an open-source game server that handles all of the environment logic for 
this project. It can be found 
[here](https://github.com/smogon/pokemon-showdown/blob/master/server/README.md) with its own 
installation instructions.

### 2. Poke-Env

Poke-Env is an open-source project on Github that handles all of the communication with Pokemon 
Showdown. This library allows us to program agents without having to worry about HTTPS protocols and
other annoying things. For setup:

```
git clone https://github.com/hsahovic/poke-env.git
cd ./poke-env
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .[all]
```

### 3. Setup Champion League

`pip install -r requirements.txt`

### 4. Scripted Agent vs. Scripted Agent

The following command will generate two scripted agents (no GPU), connect them to the Showdown 
server, and have them battle continuously.

`python -m champion_league.scripts.max_damage_battle`

### 5. Train An Agent
The following command will start a training script using PPO and self-play:

`python -m champion_league.scripts.self_play logdir={PATH} tag={NAME}`

The command first runs the script `self_play.py` in the `./champion_league/scripts/` directory. It
then configures the log directory -- where the agent's network weights, arguments, and other useful
information will be stored -- and the tag (or name) of the agent. All information on the agent is 
then stored in the directory `{PATH}/{NAME}/`.