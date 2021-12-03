import java.io.*;
import java.util.*;

/**
 * Builds a HMM from file training data and runs Viterbi on either user input or another pair of test files to tag the parts of speech of the test words
 *
 * @author Logan Chang, PS5, CS10, 20F
 * @author Ashna Kumar, PS5, CS10, 20F
 */
public class HMM {
    HashMap<String, HashMap<String, Double>> observationScores; //scores for transitions in form {tag -> {observed word -> score}}
    HashMap<String, HashMap<String, Double>> transScores;   //scores for transitions in form {tag -> {transition tags -> score}}
    BufferedReader wordIn;      //file reader for sentence train file
    BufferedReader tagIn;       //file reader for tag train file
    final double U = -100.0;    //unseen word penalty
    static Scanner scan = new Scanner(System.in);   //console input reading scanner

    /**
     * Construct a HMM object that will instantiate the Maps representing the HMM
     */
    public HMM() {
        observationScores = new HashMap<>();
        transScores = new HashMap<>();
    }

    /**
     * Parse the pair of training files and build a HMM from them
     *
     * @param wordFile Path to word file
     * @param tagFile  Path to tag file
     * @throws IOException Possible IOException when reading
     */
    public void buildHMM(String wordFile, String tagFile) throws IOException {
        //create readers for the training files
        wordIn = new BufferedReader(new FileReader(wordFile));
        tagIn = new BufferedReader(new FileReader(tagFile));
        try {
            String[] wordArray, tagArray;
            String wordLine, tagLine;
            //read in each line individually from the two training files simultaneously
            while ((wordLine = wordIn.readLine()) != null && (tagLine = tagIn.readLine()) != null) {
                wordArray = wordLine.toLowerCase().trim().split(" ");
                tagArray = tagLine.trim().split(" ");
                //create the scores from the lines being read
                createScores(wordArray, tagArray);
            }
            //normalize the scores of the HMM
            normalizeScores();
        }
        //close training files
        finally {
            wordIn.close();
            tagIn.close();
        }
    }

    /**
     * Creates the scores for the HMM observations and transitions
     *
     * @param words String array of training words
     * @param tags  String array of training tags
     */
    public void createScores(String[] words, String[] tags) {
        //loop through all words in the training (and consequently all tags)
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            String tag = tags[i];
            //mark that we have observed the given word and tag together
            markObservation(word, tag);
            //if its the first word and tag, put in a special case of '#' (start case) transitioning to the first word's tag
            if (i == 0) {
                markTransition("#", tag);
            }
            //count transitions for everything except the last string (doesn't transition anywhere)
            if (i < words.length - 1) {
                String nextTag = tags[i + 1];
                //mark tha transition from the current training tag to its following training tag
                markTransition(tag, nextTag);
            }
        }
    }

    /**
     * Mark the transition of the current tag and its word in training data
     *
     * @param word Current word
     * @param tag  Matching tag
     */
    public void markObservation(String word, String tag) {
        //if we have never seen this tag before
        if (!observationScores.containsKey(tag)) observationScores.put(tag, new HashMap<>());
        //first time seeing tag -> word
        if (!observationScores.get(tag).containsKey(word)) observationScores.get(tag).put(word, 0.0);
        //increment the score of tag -> word
        observationScores.get(tag).put(word, observationScores.get(tag).get(word) + 1);

    }

    /**
     * Mark the transition of the current tag and next tag in training data
     *
     * @param currTag Current tag
     * @param nextTag Next tag
     */
    public void markTransition(String currTag, String nextTag) {
        //if we have never seen currTag before
        if (!transScores.containsKey(currTag)) transScores.put(currTag, new HashMap<>());
        //first time seeing curr tag -> next tag
        if (!transScores.get(currTag).containsKey(nextTag)) transScores.get(currTag).put(nextTag, 0.0);
        //increment the score of currTag -> nextTag
        transScores.get(currTag).put(nextTag, transScores.get(currTag).get(nextTag) + 1);
    }

    /**
     * Normalize the scores as natural logs
     */
    public void normalizeScores() {
        //for every training tag that's been observed
        for (String tag : observationScores.keySet()) {
            //find how often we have observed this tag
            double totalFreq = 0;
            for (String word : observationScores.get(tag).keySet()) {
                totalFreq += observationScores.get(tag).get(word);
            }
            //for every word with this tag, normalize its score
            for (String word : observationScores.get(tag).keySet()) {
                observationScores.get(tag).put(word, Math.log((observationScores.get(tag).get(word)) / totalFreq));
            }
        }
        //for every training tag we've observed
        for (String tag : transScores.keySet()) {
            //find how often the current tag has had an observed transition
            double totalFreq = 0;
            for (String nextTag : transScores.get(tag).keySet()) {
                totalFreq += transScores.get(tag).get(nextTag);
            }
            //for every tag the current tag has been observed transitioning to, normalize its score
            for (String nextTag : transScores.get(tag).keySet()) {
                transScores.get(tag).put(nextTag, Math.log((transScores.get(tag).get(nextTag)) / totalFreq));
            }
        }
    }

    /**
     * Run Viterbi on the given sentence using the training-constructed HMM to predict the POS of each word in the test sentence
     *
     * @param line A test line of words (observations) that tags need to be guessed for
     * @return The best possible tags at each word as an ArrayList
     */
    public ArrayList<String> viterbi(String line) {
        String[] words = line.toLowerCase().trim().split(" ");
        ArrayList<String> path = new ArrayList<>(); //store the final tags of the sentence
        ArrayList<HashMap<String, String>> backPath = new ArrayList<>(); //store the possible tags and its previous tags at each word
        //current states and scores for each word we pass through
        List<String> currStates = new ArrayList<>();
        HashMap<String, Double> currScores = new HashMap<>();
        currStates.add("#");
        currScores.put("#", 0.0);
        //for every observed word in our test
        for (int i = 0; i < words.length; i++) {
            backPath.add(new HashMap<>());
            ArrayList<String> nextStates = new ArrayList<>();
            HashMap<String, Double> nextScores = new HashMap<>();
            //for all possible current states
            for (String currState : currStates) {
                if (transScores.get(currState) != null) {   //safety check against some extraneous cases (punctuation etc.)
                    //for all possible states the current state could transition to based on observed training transitions between POS
                    for (String nextState : transScores.get(currState).keySet()) {
                        if (!nextStates.contains(nextState)) nextStates.add(nextState);
                        double obsScore;
                        try {
                            obsScore = (observationScores.get(nextState).get(words[i]));
                        } catch (NullPointerException e) {
                            obsScore = U;
                        }
                        //get its transition score to this next possible state
                        double nextScore = currScores.get(currState) + transScores.get(currState).get(nextState) + obsScore;
                        //if it is the highest score we have seen for this word, specifically for currTag -> nextPossibleTag, save it
                        if (!nextScores.containsKey(nextState) || nextScores.get(nextState) < nextScore) {
                            nextScores.put(nextState, nextScore);
                            backPath.get(i).put(nextState, currState);
                        }
                    }
                }
            }
            //set the next states and scores to our current data as we step forward
            currStates = nextStates;
            currScores = nextScores;
        }
        //get the highest score of the final scores once we have read all the input words
        double maxScore = Integer.MIN_VALUE;
        String bestLastTag = "";
        for (String tag : currScores.keySet()) {
            if (currScores.get(tag) > maxScore) {
                maxScore = currScores.get(tag);
                bestLastTag = tag;
            }
        }
        path.add(bestLastTag);
        //back track from the best possible final tag to get all previous best possible tags
        int pos = words.length - 1;
        String prevTag = backPath.get(pos).get(bestLastTag);
        while (!prevTag.equals("#")) {
            path.add(0, prevTag);
            bestLastTag = prevTag;
            pos--;
            prevTag = backPath.get(pos).get(bestLastTag);
        }
        return path;
    }

    /**
     * Run Viterbi on test file and compare its score to the test files tags
     *
     * @param wordFile Test file of words/sentences
     * @param tagFile  Test file of correct tags
     * @throws IOException Possible IOException when reading test files
     */
    public void testOnFiles(String wordFile, String tagFile) throws IOException {
        //readers for test files
        BufferedReader testIn = new BufferedReader(new FileReader(wordFile));
        BufferedReader testTagIn = new BufferedReader(new FileReader(tagFile));
        try {
            String testLine;
            int good = 0, bad = 0;
            //compare what viterbi predicts with the actual tags for each line in the test file
            while ((testLine = testIn.readLine()) != null) {
                String testTagsLine = testTagIn.readLine();
                ArrayList<String> path = viterbi(testLine);
                String[] testTags = testTagsLine.split(" ");
                System.out.println("\n"+testLine);
                StringBuilder sb = new StringBuilder();
                for(String tag: path){
                    sb.append(tag+" ");
                }
                System.out.println("=> "+sb.toString().strip());
                for (int i = 0; i < testTags.length; i++) {
                    if (testTags[i].equals(path.get(i))) good++;
                    else bad++;
                }
            }
            //print the results of tags correctly and incorrectly matched with test tags file
            System.out.println("Tagged correctly: " + good + "\tTagged Incorrectly: " + bad);
        }
        //close test files
        finally {
            testIn.close();
            testTagIn.close();
        }
    }

    /**
     * Run Viterbi built on training files on user-input sentences/words in the console
     */
    public void testOnInput() {
        String line;
        while (true) {
            System.out.println("Enter your sentence (type 'stop' to end reading): ");
            line = scan.nextLine();
            if(line.equals("stop")) break;
            StringBuilder sb = new StringBuilder();
            for(String tag: viterbi(line)){
                sb.append(tag+" ");
            }
            //print the input line and the viterbi tags of that line based on training data HMM
            System.out.println("\n"+line.toLowerCase());
            System.out.println("=> "+sb.toString().strip());
        }
    }

    /**
     * DRIVER CODE
     *
     * Please change the path name variables for each file!
     *
     * @param args Command-line arguments (ignored)
     * @throws IOException Possible IOException when opening/reading files
     */
    public static void main(String[] args) throws IOException {
        //EDIT THESE FOR EACH NEW TRAINING/TESTING FILES
        String trainSentencesPath = "inputs/simple-train-sentences.txt";
        String trainTagsPath = "inputs/simple-train-tags.txt";
        String testSentencesPath ="inputs/simple-test-sentences.txt";
        String testTagsPath ="inputs/simple-test-tags.txt";
        HMM hmm = new HMM();
        boolean readFromFile = false;
        hmm.buildHMM(trainSentencesPath, trainTagsPath);
        System.out.println("Read from file or console? ('f' for file, 'c' for console): ");
        String read = scan.nextLine();
        if (read.equals("f")) readFromFile = true;
        if (readFromFile) hmm.testOnFiles(testSentencesPath, testTagsPath);
        else hmm.testOnInput();
        scan.close();
    }
}
