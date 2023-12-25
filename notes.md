# Notes for HW1 

## Debugger
if we are using debugger from vs code with argus. We can use lauch.json file to add arguments parameters. And for the args and values, they are in sinlge "" at a time.

```
Such as :
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": ["--expert_policy_file",  "cs285/policies/experts/Ant.pkl", "--env_name", "Ant-v4", "--exp_name", "bc_ant", "--n_iter", "1", "--expert_data", "cs285/expert_data/expert_data_Ant-v4.pkl", "--video_log_freq", "-1"]
        }
    ]
}
```


## TODO:
MLP turns out can be learn by probability distributions. But currently, it doesn't work in my case. I added two kinds of implementation, one will use a normal MLP to learn the actions along with the observation, the other will learn a normal distribution. 

The distribution method, it turns out the logstd can not be learned. The values in logstd has not change. I wonder if the require_grad is working or not. It's very strange. I guess I should investigate how to cretea a distribution and learn it. 

The next day will be DAgger.

## Done
In Dec 25th, I have added the sample trajectories function, the behavior cloning function, MLP_policy class and the run_hw1.py. 

