public class RandomAgent implements Agent {
    
    private double[] action;

    /**
     * Provides the task specification to the agent and initializes the agent.
     * This method is called once at the start of each session, and should be
     * used to perform any one-time initialization tasks, such as constructing
     * data structures or reading training data from a file. The last parameter
     * is given the value of the "-q" or "--agent-param" option that was passed
     * at the command line. It can be used to provide additional custom
     * initialization information to the agent, such as the name of a file to
     * read training data from. It will be null if nothing was supplied.
     *
     * This method return must a name for the agent, which is used to
     * identify the agent in log files produced by the environment. The name
     * must contain only letters, digits, and underscores.
     *
     * @param stateSize the dimensionality of the state space.
     * @param actionSize the dimensionality of the action space.
     * @param param an custom initialization parameter.
     * @return the agent's name.
     */
    public String init(int stateSize, int actionSize, String param) {
        action = new double[actionSize];
        return "JavaRandomAgent";
    }
    
    /**
     * Performs and returns the first action of an episode.
     *
     * @param initialState the initial state of the episode.
     * @return the agent's first action.
     */
    public double[] start(double[] initialState) {
        newAction();
        return action;
    }
    
    /**
     * Performs one step of the agent.
     *
     * @param state the current state.
     * @param reward the reward resulting from the agent's last action.
     * @return the agent's next action.
     */
    public double[] step(double[] state, double reward) {
        newAction();
        return action;
    }
    
    /* This method is specific to the RandomAgent. It is not part of
     * the Agent interface. */
    private void newAction() {
        for (int i = 0; i < action.length; i++) {
            action[i] = Math.random();
        }
    }
    
    /**
     * Processes the final reward of an episode.
     *
     * @param the final reward of the episode.
     */
    public void end(double reward) { }
    
    /**
     * Performs any finalization tasks, such as releasing resources or
     * writing training data to a file. This method is called once at the end
     * of each session.
     */
    public void cleanup() { }
}
