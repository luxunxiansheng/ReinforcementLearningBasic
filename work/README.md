<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github/luxunxiansheng">
    <img src="images/dino.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Reinforcement Learning Easy Playgroud</h3>

  <p align="center">
    A lightweight framework of reinforcement learning built for easy to learn and use  
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Background](#background)  
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About The Project

[Reinforcement Learning: An introduction 2e](1) is believed to be the most famous book on reinforcement learning methodology. Dozons of classic algorithms are proposed in this book, which include dynamical progamming methods, monte carlo methods, temporal-difference  methods,policy grident methods and so on so forth.  [Shangtong Zhang](2) implements almost all of the algothims in the book with python and it is my favorite reference project when I read Sutton's book. However, because of its educational purpose, each case follows the book pretty strictly and hard to reuse its codes. From enginerring perspecitve, there is still big room for improvement. Sometimes,I need one compact code set for my own use. That is why I decided to launch this project even though I knew there had been so many others exsiting so far.

In this project,a lightweight framework is configured with Object-oriented programming paradigm. Algorithms and enviroments are carefully seperated and most of the algorithms in [Reinforcement Learning: An introduction 2e](1) are re-implemented. It might be a good start point for beginners to step into the door of reinforcemet learning, espcially for those who come from software industry with solid programming skills.

Furthermore, deep inforcement learning algorithms,such as DQN, Alphazero,etc., will also be covered.

This project is built With [Python](www.python.org) and several libraries:

* [gym](http://openai.com)
* [tqdm](http://github/tqdm/tqdm)


## Background

### 1. The differences between RL and supervised learning, unsupervised learning

   General speaking, machine learning is roughly categoried into three paradigms: supervised learning, unsupervised learning as well as reinforcement learning. A fundamental characteristic for both supervised and unserpervised learning is that their learning machineries are based on  exsiting datasets while reinforcement learning's is based on the enviroment.  In other words, when the supervised and unsupervised learning come into play, the dataset is avaliable anyway, but for reinforcement learning, it is the responsibility of the agent to acquire the dataset by interact with the enviroment. The purpose of supervised learning and unsupervised learning is to figure out what the world looks like while reinforcement learning is to find the best way to get the most rewards instead of fitting the enviroment.
  
### 2. The two steps of RL

   Almost all of the algothims of the RL consist of two parts: 1. generate data; 2. find the optimal policy based on those generated dataset. We will not only take the complexity of the algorithm into account , but also concern how the data is generated in term of 'Data Efficiency'.  In a nutshell,  the challenge of RL is how to improve the data effciency as well as optimize the algoritms to achieve the total rewards.

### 3.[Exploration-Exploitation Dilemma](https://zhuanlan.zhihu.com/p/161284124)

   In RL, since what we are dealing with is enviroment rather than given dataset, it is free for agent to get desired data to learn and then to improve the policy via those data. The data quality is critical. On one side, it is better to make a good use on the collected data ; on the other hand, it is better not to miss more valubale data. The exploration and exploitation banlance is the core of the RL.

   Basically, there are two types of cases in RL about the exploration-exploitation dilemma. In the first case, trainning and testing proceed in an interlaced manner. In this scenario, the purpose of exploration-exploitation is to get as many rewards as it can. In the second case, there are solely two stages: tranning and testing somehow as supervied learning does. The purpose of exploration-exploitation during the tranning is not to get as much rewards as it can, but to keep the distribution of state-action pair in tranining set as close as the optimal actions in the current stage so far.

   Briefly speaking, two things to do, 1. Based on the current collected dataset, to see  which state-action pairs look better and then to give those good candidates more weights to try to get more accurate estimatiion(Exploitation). 2. Based on the new more accurate estimation,improve the policy and to collect data which looks more promising but without ingoring those which looks not much promising in case the local convergence occurs (Exploration) and 3. go to the next loop.

### 4.[The primary ideas on value based RL and policy based RL](https://zhuanlan.zhihu.com/p/165295616)

   1. Value based RL is following a dynamic programming methodology. That is, to seperate a big problem into many small ones;Policy based RL  only concerns a global policy. 

   2. Value based RL could essentially be regarded as solve the Bellman Equation; Policy based RL is more like an optimation process to get the maximum utility or minmum losses.

### 5.On-Policy and Off-policy

   1. Once generating the new data, we will use it immeditely to train the learner,which is so called on-policy. If we store the generated data in the cache for the later use, it is off-policy.

   2. For off-policy, the expectation of Q(s,a) is to the enviroment; while on-policy is about enviroment and the policy. <br> 

   > For example: <br>   
   >
   > $Q-learning:  Q^*(s,a)= E_{envrionment}(R_s^a+\gamma{max_{a'}Q^*(s',a')})$
   >
   > $Sarsa:       Q(s,a)= E_{envrionment,\pi}(R_s^a+\gamma{Q_\pi(s',a')})$  

### 6. How to avoid local convergence

   1. More training data 
   2. Increase the exploration 
   3. Expericence replay 
   4. More dense reward


## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
3. Install NPM packages
```sh
npm install
```
4. Enter your API in `config.js`
```JS
const API_KEY = 'ENTER YOUR API';
```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)



## References
* [1] http://incompleteideas.net/book/the-book-2nd.html 
* [2] https://github/ShangtongZhang/reinforcement-learning-an-introduction 

