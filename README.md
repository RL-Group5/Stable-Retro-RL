# Stable-Retro-RL

_In environments with complex dynamics and high-dimensional action spaces,
learning effective policies remains a challenging task. Motivated by the need to
balance exploration, temporal abstraction, and decision-making efficiency, we
explore a hierarchical approach to reinforcement learning that assigns diverse
roles to specialized workers operating at multiple levels of abstraction. This work
presents a comparative study between Proximal Policy Optimization (PPO)—a
well-established, on-policy actor-critic method—and Hierarchical Proximal Policy
Optimization (HPPO), which augments PPO with a structured hierarchy to bet-
ter capture long-term dependencies and improve exploration. By disentangling
decision-making across temporal scales, HPPO aims to address limitations of PPO
in scenarios requiring sustained strategic planning. To evaluate these methods, we
employ Mortal Kombat II within the stable-retro framework, a visually rich and
action-dense environment. Additionally, we implement parallelized training to
ensure scalability and consistent learning across both approaches._

https://github.com/user-attachments/assets/287662f5-7f15-4884-9feb-cc0aa19f0ce2

Below we have the rewards from PPO vs HPPO. We can clearly see how well HPPO Learns and 
how stable it is compared to PPO.
![Image](https://github.com/user-attachments/assets/e5df1d88-a9f0-4250-85e0-c1bcdd97cfc4)

![image](https://github.com/user-attachments/assets/3c02fb68-2351-4ac2-8f8f-ff5472428f13)
