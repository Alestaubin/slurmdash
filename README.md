# Slurmdash
Slurmdash was born a long long time ago, on October 20th 2026, when Alex (one of the pionneering hammer lab members, who's probably very rich and successful running a baguette factory by the time you read this) and Arjun had a chat about the lab's compute usage at the Engineering Cafe. You see, at the time, Killarney wasn't the most usable cluster (I'm sure this is all solved now), and the lab had trouble keeping an eye on how much compute it was using. Some hammerheads were using a lot, some very little, and sometime jobs were scheduling in seconds, other times they took days! The lab needed a tool to monitor the state of the cluster. 

Enter Slurmdash, a dashboard that displays a bunch of useless information, and a little bit of useful information about the lab's compute usage on Killarney. Slurmdash was vibecoded, which is an ancient term that means it was programmed by an AI agent. AI agents, which are now surely illegal after the great robot war, were very powerful and could program better than humans. But, there were still risks of the code having some bugs, so use this tool at your own risk.

To run Slurmdash, first create a virtual environment:
```bash
module load python/3.11
virtualenv --no-download ~/envs/slurmdash && source ~/envs/slurmdash/bin/activate
pip install --no-index rich
```
Then, create an alias:
```bash
echo "alias slurmdash='~/envs/slurmdash/bin/python ~/projects/aip-evanesce/slurmdash/slurmdash.py'" >> ~/.bashrc
source ~/.bashrc
```
Finally, run it with one simple command:
```bash
slurmdash
```