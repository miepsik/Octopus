
               Octopus Arm Simulator
               =====================

Last changed: 2006-05-02

Contents:
=========

     I. OVERVIEW AND REQUIREMENTS

    II. WRITE AND COMPILE YOUR AGENT

   III. RUNNING THE ENVIRONMENT

    IV. COMMUNICATION PROTOCOL DOCUMENT 

     V. ENVIRONMENT PHYSICS DOCUMENT
     
    VI. LOG OF THE REWARDS OBTAINED

   VII. KNOWN BUGS (none currently)


============================
I. OVERVIEW AND REQUIREMENTS
============================

This document should help you write and compile an agent that will run in the octopus environment.

The environment is written in Java and communicates with the agent over sockets, using a protocol following the lines of the RL-Glue concepts. In principle, this allows agents to be written in any language and run on any platform. However, we do not expect you to be familiar with socket programming. For this reason, we provide "agent handlers", which act as intermediaries between your agent and our environment, that completely abstract you from the socket communication.

Running the environment requires that version 5.0 or later of the Java Runtime Environment (JRE) be installed on your system. To verify this, type "java -version" at a command prompt.  If you do not have the JRE, or do not have the correct version, you can download it for free by going to http://java.sun.com/j2se/1.5.0/download.jsp, clicking on "Download JRE 5.0 Update 6", and following the instructions presented.


====================================
II. WRITING AND COMPILING YOUR AGENT
====================================

Currently, our agent handlers support C (and C++ by extension), Java and Python. For convenience, we provide "template agents" in all supported languages, which perform random actions; you can modify these to implement your agent. For convenience, every agent must have a name which it must pass to the environment that will be used to name the log file outputed by the environment containing the rewards obtain over the episodes.


II.A   C Agents
---------------

In C, all agents must implement the function prototypes given in agent.h, located in "agent/c" directory of the distribution. You will also find a template agent, template_agent.c, which you can modify to write your agent.

The provided Makefile was tested on linux and windows (cygwin). We also provided a Makefile.solaris if your uname is SunOS. If there are a lot of requests, we may provide a Windows (Visual Studio) version.

To run the agent, you need to specify the host, port, number of episodes and (optionaly) parameters that will be used by your agent. For example:

./agent_handler localhost 10000 10

Type "./agent_handler" with no additional arguments for a description of available options.


II.B   Java Agents
------------------

In Java, all agents must implement the Agent interface; consult the template agent, contained in the file RandomAgent.java in the "agent/java" directory of the distribution, for a list and description of the methods in the interface. When compiling your agent, you must include java-agent-handler.jar (also in the "java" directory) on the classpath when compiling your agent. For example:

javac -cp java-agent-handler.jar MyAgent.java

Note that there is no need to open java-agent-handler.jar, or to compile any source files other than those of your agent.

Once you have compiled your agent and started the environment, you can run it by executing java-agent-handler.jar, providing the class name of your agent and the environment's port as command-line arguments. For example:

java -jar java-agent-handler.jar -a MyAgent -p 1050.

Type "java -jar java-agent-handler.jar" with no additional arguments for a description of available options. The -c (agent classpath) option is especially useful if your agent is located in a JAR file, or uses additional libraries.


II.C	Python agents
---------------------

In Python, all agents must implement the public (i.e. those not starting with "__") methods of the template agent, agent.py, which you can find in the "agent/python" directory of the distribution. To run the agent, you need to call agent_handler.py, which will handle the communication with the environment and call the appropriate methods of your agent. For example:

python agent_handler.py 127.0.0.1 10000 1

Type "python agent_handler.py" with no additional arguments for a description of available options.


============================
III. RUNNING THE ENVIRONMENT
============================

To run the environment, execute octopus-environment.jar, located in the "environment" directory of the distribution. The specific command-line syntax differs depending on the execution mode selected. Three modes are available:

  - Internal control: you control the octopus arm with the keyboard, and can graphically see the environment state.
  - External control with graphical display: your agent controls the arm, and you can graphically see the environment state.
  - External control without graphical display: your agent controls the arm, and no graphical output is given. This is the preferred mode for agent learning.

These are further described in the following subsections.

In any of the three modes, an XML settings file, which contains the initial state, physical parameters, and a specification of the agent's task. A recommended settings file is supplied in the root folder of the distribution.


III.A  Internal control
-----------------------

To run the environment with internal control, type

java -jar OctopusEnvironment.jar internal settings.xml

Only a limited subset of the (continuous) action space is available with internal control. Specifically, six keys are mapped to different "primitives" from which actions can be constructed:

 - Z: fully contract all dorsal muscles on the lower half of the arm
 - X: fully contract all transversal muscles on the lower half of the arm
 - C: fully contract all ventral muscles on the lower half of the arm
 - I: fully contract all dorsal muscles on the upper half of the arm
 - O: fully contract all transversal muscles on the upper half of the arm
 - P: fully contract all ventral muscles on the upper half of the arm
 
Multiple keys may be pressed simultaneously, allowing for a total of 2^6 discrete actions.

There is currently no function to start a new episode within the graphical interface; you must close and restart the program.


III.B  External control with graphical display
----------------------------------------------

To run the environment with external control and a graphical display, type

	java -jar OctopusEnvironment.jav external_gui settings.xml <PORT_NUMBER>

Your network and operating system may impose restrictions on what ports you can successfully use. Make sure that your OS allows you to start a server on the port you specify, and make sure the port is open to the agent that will connect to it, particularly if the agent and environment are not co-located.

This mode is useful to visually validate what your agent is doing, but it is unsuitable for learning, as the simulation speed is reduced by graphical rendering as well as delays between animation frames. Use the next mode for learning.


III.C  External control without graphical display
-------------------------------------------------

To run the environment with external control and no graphical display, type

	java -jar OctopusEnvironment.jav external settings.xml <PORT_NUMBER>
    
This mode is preferred when agents are learning, since the simulation occurs at maximum speed.

To stop the environment, you must forcibly terminate its process using the functions provided by your operating system.


===================================
IV. COMMUNICATION PROTOCOL DOCUMENT
===================================

A document describing the protocol of communication between the agent handler and the environment will be made available at the distribution site in the near future. Note that if you are using the handlers we provided, you do not need to read this document.


===============================
V. ENVIRONMENT PHYSICS DOCUMENT
===============================

A document explaining the environment's physics is available at the distribution site.


===============================
VI. LOG OF THE REWARDS OBTAINED
===============================

For benchmarking purposes, the environement saves a file "NameOfAgent-Date.log" which contains the reward value at each step. Each episode is on a different line and the rewards of each step of an episode are separated by a single space.


===============
VII. KNOWN BUGS
===============

There are no known bugs as of 2006-05-02. If you notice any unexpected behaviour, please report to dan.cast@mail.mcgill.ca.