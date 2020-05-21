---
layout: post
title:  "Do you need feedback from your dev-team?"
description: "...and why you should push your team to implement a code review system!"
---

## Why feedback?

When the deadlines are approaching, the time used to think and test is reduced. Pressure raises and it is more easy to make mistakes, creating bugs in the code.

Fixing a bug during the design phase is always cheaper than fixing a bug in QA or when the code is in production.

Teams should have mechanisms in place to fix bugs before they advance towards production. One important mechanism is feedback.

## What is feedback?

Feedback is a process that continuously compares the actual output to its desired reference value, and applies changes to reach it. 

Many many many many companies today still operate using the following rigid scheme:

1. System requirements
2. Sofware requirements 
3. Analysis <- this is supposed to be feedback in theory
4. Program design
5. Coding
6. Testing  <- this is supposed to be feedback in theory
7. Ship to Production

This is not efficient and against the feedback principle. There is no feedback. It is just a cascade of actions that underestimate the intrisic mutability of the codebase of each project.

When a developer obtains feedback quickly and constantly, grows and learns. If code with no review is pushed into the codebase the cost of fixing a bug might be extremely high and in some cases (industrial code), create some phisical damage.


## How to implement feedback in your team?

Feedback is an iterative scheme, constantly adjusting the output. Some suggestions are:

- Pair programming,
- Agile development,
- Code reviews/Pull requests,
- Code style guides,
- Unit tests. 

On top of reducing inefficiencies and improving code quality, junior developers on the projects will gain extremely valuable feedback that will have positively furthered their education and career.

In the following I will discuss two points that are very important: code reviews and **code reviews!**

## Code Reviews

Code review is the core habit that your team should form in order to have an efficient feedback loop. It is not only the pillar of your team but the pillar of a sustainable codebase. Code reviews help teams, within the organization, to have consistent quality of projects across the board.

During code reviews bug can be spotted. Repetitive code can be seen and eliminated (DRY). Important questions can be answered: 
- Does it meet the requirements? 
- What about the architecture? Is it solid? Is it future-proof? 
- Can this be reused somewhere else or in another project?
- etc.etc.

Code reviews not only reduce bugs in the code but more importantly spread code ownership and mentor developers when they join the team.

If only few seniors in the project do code reviews, it is **A REALLY BAD HABIT!**
This distribute knowledge about the overall architecture only to the seniors. What about if the senior devs quit? Outsourcing code reviews to external senior dev is also a bad idea, especially if there is a lack of documentation or the software requirements are unclear. 

Code review is a task for everyone within the team!

It is also important to underline that to have efficient code-review system takes effort.

Pull requests are a modern tool to move away from the previous old rigid scheme, previously discussed 👆. They provide a better way of reviewing the code base, tranforming the review in an asynchronous, distributed, social, trackable, public task.


## Pull Requests
If you are a software developer and you have lived under a rock you are justified if you don't know what version control is. Actually, you are not!💣

### What is git?
[Git](https://git-scm.com/) is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency.

Git is easy to learn and has a tiny footprint with lightning fast performance. It outclasses SCM tools like Subversion, CVS, Perforce, and ClearCase with features like cheap local branching, convenient staging areas, and multiple workflows.

### What are pull requests?
Pull requests are a very good form of continuous code review. They are actually a bit more than just code review. 

Pull Requests are a feature of some web-based version control repositories like [GitHub](https://github.com/) or [BitBucket](https://bitbucket.org/).

Pull Requests are commonly used by teams and organizations collaborating using the Shared Repository Model, where everyone shares a single repository and topic branches are used to develop features and isolate changes. Many open source projects use pull requests to manage changes from contributors as they are useful in providing a way to notify project maintainers about changes one has made and in initiating code review and general discussion about a set of changes before being merged into the main branch.

"Pull requests let you tell others about changes you've pushed to a code repository. Once a pull request is sent, interested parties can review the set of changes, discuss potential modifications, and even push follow-up commits if necessary.
After initializing a pull request, you'll see a review page that shows a high-level overview of the changes between your branch (the compare branch) and the repository's base branch. You can add a summary of the proposed changes, review the changes made by commits, add labels, milestones, and assignees, and @mention individual contributors or teams.
Other contributors can review your proposed changes, add review comments, contribute to the pull request discussion, and even add commits to the pull request."

It is important to have one issue  in one pull request, so you can reason about why the changes were made in the branch.

Every merge should be approved minimum by two reviewers to four reviewers. Why? Limiting the reviewers to a small group improves productivity, otherwise, reviewers will just skim the 5000 lines you have sent thinking that another reviewer has done a better job. Wrong!


### Why are pull requests great?

They provide a better code review culture in general. How?

- Making the review activity visible
    + Review activity should be easily consumable (activity stream)
- Making code-sharable
- There is collaboration on the review
- The code lives in ONE common place, not in e-mail attachments
- Discussions are encouraged
- Changes are historically traceable


## It takes too much time...

Senior oversight on projects is really important. A common pitfall is that some seniors are __too busy__ to review code.

The time spent on code reviews **will pay back** in time not spent dealing with angry customers, as well as not dealing with the expensive task of fixing bugs in production.

Code reviews are more important than deadlines❗ They have to be included ahead from the project managers and requested from all the developers on the team.

## Some additional tips

Mentorship is an important **responsibility**. Lead developers have the responsibility to review. Junior developers have the responsibility to ask for a review and follow up with questions trying to gain as much as possible from the feedback.

- Give high priority to the most sensitive/scary parts, especially the code that nobody knows
- It is important to find problems and not write solutions in the code review. This is a good opportunity to pair program.
- Huge methods need to be reduced to multiple methods with one single task (Unix philosophy)

