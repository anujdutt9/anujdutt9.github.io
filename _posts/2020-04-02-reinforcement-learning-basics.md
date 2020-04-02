---
layout: post
title: 'Reinforcement Learning-The Basics'
tags: [Reinforcement-Learning]
featured_image_thumbnail: assets/images/posts/2020/reinforcement-learning/rl-basics.jpeg
featured_image: assets/images/posts/2020/reinforcement-learning/rl-basics.jpeg
featured: true
hidden: true
---

Hello everyone. Welcome to the first blog in this series on Reinforcement Learning. In this tutorial we’ll start by understanding what is Reinforcement Learning and how it is different from other categories in machine learning. Then we’ll define the main parts of any Reinforcement Learning system like an agent, environment, states & actions and observations.
So, let’s get started.

# Definition

>Reinforcement Learning is a subfield of Machine Learning that deals with automatic learning of optimal decisions over time.

Let’s understand this with the help of an example and also see what is the difference between this and the other fields of machine learning.
 
Let us consider the game of chess. We want to train a system/model to be able to learn the rules of chess, as well as play it, just like a professional. In this case, we cannot make use of  supervised learning techniques as there is no correct label w.r.t image of the current state of the chessboard. In the game of chess, you either win or lose the game and that could take a lot of moves. Hence, <mark>a label does not exist for every move to classify the images into a win or a loss.</mark>

{% include image-caption.html imageurl="/assets/images/posts/2020/reinforcement-learning/chess.jpg" title="Chess Board" caption="Chess Board" %}

On the other hand, this is not a totally unsupervised system where you could apply unsupervised learning techniques as the system/model gets some kind of a reward at the end of the game i.e. win or lose the game.

From the above discussion we see that the techniques for supervised and unsupervised learning are not directly applicable to our problem. This is where reinforcement learning comes in. 

<mark>Reinforcement Learning lies in between supervised and unsupervised learning methods.</mark> In reinforcement learning, we have an **agent** [system/model] that interacts with the **environment** [chess in this case] using the channels of **actions**, **reward** and **observations**.

Let’s define all these terms one by one.

# Agent
An agent is someone or something that interacts with the environment by taking an action, making observations and eventually recieving a reward. For example, in our chess game, the player or the computer program is the agent that tries to solve the problem in a more or less efficient way.

# Environment
The environment is everything apart from the Agent. For example, in the game of chess, the chessboard and the player/agent in the opposition is the environment.

{% include image-caption.html imageurl="/assets/images/posts/2020/reinforcement-learning/RL-System.png" title="Chess Reinforcement Learning World" caption="Chess Reinforcement Learning World" %}

# Actions
Action is something the agent can do in an environment. For example, in the game of chess the actions are all the actions that a particular character can make like move forward, backward, left, right etc.

# Reward
In the field of Reinforcement Learning, a reward is a scalar value that is given periodically to the agent as it interacts with the environment. This reward could be positive, negative or a neutral value, could be a small or a large value as well as could be provided to the agent at every timestamp or at the end.
<mark>The aim of reward is to reinforce the behavior of the agent in the environment so that it can learn to achieve its goals.</mark>

# Observations
Observations are something that the environment provides to the agent telling the agent about what’s going on around it in the environment. For example in the game of chess, the observation is your current position on the chessboard.
The observations can even include reward in some vague form. For example, the pixels of the scorecard on the screen of a game.

{% include image-caption.html imageurl="/assets/images/posts/2020/reinforcement-learning/score-pixels.jpg" title="Atari Game with Scorecard" caption="Atari Game with Scorecard" %}

# Example

Let us bring together all the concepts learnt so far using an example:

**Statement:** "Athelete runs on the track."

In this statement, <br/>

<table>
<thead>
<tr>
   <th>Statement</th>
   <th>RL World</th>
  </tr>
</thead>
 <tbody>
  <tr>
   <td>Athelete</td>
   <td>
    Agent<br />
   </td>
  </tr>
  <tr>
   <td>Running</td>
   <td>Action</td>
  </tr>
  <tr>
   <td>Running Track, Stadium</td>
   <td>Environment</td>
  </tr>
 </tbody>
</table>

{% include image-caption.html imageurl="/assets/images/posts/2020/reinforcement-learning/athelete-on-track.jpeg" title="Atheletes running on the track (Credits: Jonathan Chng)" caption="Atheletes running on the track (Credits: Jonathan Chng)" %}

Similarly, <br/>

**Statement:** "Athelete wins the prize."

<table>
<thead>
<tr>
   <th>Statement</th>
   <th>RL World</th>
  </tr>
</thead>
 <tbody>
  <tr>
   <td>Athelete</td>
   <td>
    Agent<br />
   </td>
  </tr>
  <tr>
   <td>Prize</td>
   <td>Reward</td>
  </tr>
  <tr>
   <td>End of Race/Track</td>
   <td>Observations</td>
  </tr>
 </tbody>
</table>

Since,every athelete running in the race wants to win the race, hence, getting the **reward** in terms of the prize reinforces them as **agents** to run as fast as possible.

# Summary

So, let’s summarize all these concepts together.
The agent takes an action in the environment. For every action, the agent recieves a reward and some observations. The reward could be given at every timestamp (as in a shooter game where killing a zombie increases the score) or at the end (as in the game of chess i.e. win or lose). The environment also provide the observations along with the rewards that tell what is going on around the agent in the environment.

Great work at completing this tutorial. Now that we know some basics about Reinforcement Learning, let’s move to our next topic: **The Markov Process.**
