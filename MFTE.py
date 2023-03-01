import re
from pathlib import Path
import glob
import os
import pandas as pd
import collections
import emoji
import sys
import os
import inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
#from stanza.server import CoreNLPClient
import stanza
#from stanza.pipeline.core import DownloadMethod
import multiprocessing
import timeit
import tqdm


def tag_stanford (dir_nlp: str, dir_in: str, dir_out: str) -> None:
    """Tags text files in dir_in with CoreNLPClient and writes to dir_out
    Args:
        dir_nlp (str): Stanfrod Core NLP path
        dir_in (str): dir with plain text files to be tagged
        dir_out (str): dir to write Stanford Tagger tagged files
    """
    Path(dir_out).mkdir(parents=True, exist_ok=True)   
    #text = open(dir+"corpus\BD-CMT274.txt").read()
    files = glob.glob(dir_in+"*.txt")
    if len(files) > 0:
        with stanza.server.CoreNLPClient(annotators=['tokenize,ssplit,pos'],
                timeout=30000,
                memory='6G',
                classpath= dir_nlp +'*',
                max_char_length=10000000,
                logging_level = 'INFO',
                verbose = 'False', 
                be_quiet = 'True',) as client:
            for file in files:
                text = open(file=file, encoding='utf-8', errors="ignore").read()
                file_name = os.path.basename(file)
                print("Stanford tagger tagging:", file)
                ann = client.annotate(text)
                s_list = list()
                for s in ann.sentence:
                    words = []
                    for token in s.token:
                        words.append(token.word + '_' + token.pos)
                    s_words = " ".join(words)
                    s_list.append(s_words)
                s = "\n".join(s_list)
                with open(file=dir_out+file_name, encoding='utf-8', mode='w') as f:
                    f.write(s)
    else:
        print("No files to tag.")

def stanza_pre_processing (text: str)-> str:
    """Applies preprocessing on text string before tagging it with stanza
    Args:
        text (str): Text file read from the tag_stanford_stanza function As a string
    Returns:
        text (str): Text after applying preprocessing (mainly regular expression find and replace)
    """
    #Split cannot to 2 words for better tagging with stanza
    text = re.sub("\\bcannot\\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub("\\bgonna\\b", "gon na", text, flags=re.IGNORECASE)
    text = re.sub("\\bwanna\\b", "wan na", text, flags=re.IGNORECASE)
    text = re.sub("\\bisn('|’)?t\\b", "is n't", text, flags=re.IGNORECASE) # isn't
    text = re.sub("\\baren('|’)?t\\b", "are n't", text, flags=re.IGNORECASE) # aren't
    text = re.sub("\\bweren('|’)?t\\b", "were n't", text, flags=re.IGNORECASE) # weren't
    text = re.sub("\\bdon('|’)?t\\b", "do n't", text, flags=re.IGNORECASE)
    text = re.sub("\\bwon('|’)?t\\b", "wo n't", text, flags=re.IGNORECASE)
    text = re.sub("\\bcan('|’)?t\\b", "ca n't", text, flags=re.IGNORECASE)
    text = re.sub("\\bdidn('|’)?t\\b", "did n't", text, flags=re.IGNORECASE)
    text = re.sub("\\bthat('|’)?s\\b", "that 's", text, flags=re.IGNORECASE)
    text = re.sub("\\bwhat('|’)?s\\b", "what 's", text, flags=re.IGNORECASE)
    text = re.sub("\\bit('|’)s\\b", "it 's", text, flags=re.IGNORECASE)
    text = re.sub("\\bi('|’)m\\b", "I 'm", text, flags=re.IGNORECASE)
    text = re.sub("\\byou('|’)?re\\b", "your 're", text, flags=re.IGNORECASE)
    text = re.sub("\\bhe('|’)?s\\b", "he 's", text, flags=re.IGNORECASE)
    text = re.sub("\\bshe('|’)?s\\b", "she 's", text, flags=re.IGNORECASE)
    text = re.sub("\\bthey('|’)?re\\b", "they 're", text, flags=re.IGNORECASE)
    text = re.sub("\\bi('|’)?ve\\b", "I 've", text, flags=re.IGNORECASE)
    text = re.sub("\\bwe('|’)?ve\\b", "we 've", text, flags=re.IGNORECASE)
    text = re.sub("\\bhe('|’)?d\\b", "he 'd", text, flags=re.IGNORECASE)
    text = re.sub("\\bshe('|’)?d\\b", "she 'd", text, flags=re.IGNORECASE)
    text = re.sub("\\bi('|’)?ll\\b", "I 'll", text, flags=re.IGNORECASE)
    text = re.sub("\\bhe('|’)?ll\\b", "he 'll", text, flags=re.IGNORECASE)
    text = re.sub("\\bshe('|’)?ll\\b", "she 'll", text, flags=re.IGNORECASE)
    text = re.sub("\\bit('|’)?d\\b", "it 'd", text, flags=re.IGNORECASE)

    #replace newlines with spaces within a sentence
    text = re.sub("(\w+) *[\r\n]+ *(\w+)", "\\1 \\2", text, flags=re.IGNORECASE)
    #add space between two emojis thanks to https://stackoverflow.com/questions/69423621/how-to-put-spaces-in-between-every-emojis
    text = ''.join((' '+c+' ') if c in emoji.EMOJI_DATA else c for c in text)
    return text

def process_files_list_chunk_for_stanza(files: list, nlp, dir_out: str) -> None:
    """Gets files list chunk from tag_stanford_stanza and tags with stanza nlp client and writes to dir oupt

    Args:
        files (list): list of files which needs tobe tagged
        nlp: Stanza nlp client
        dir_out (str): Output directory
    """
    print("Stanza tagger reading all files")
    #batch processing of documents, 1st list of documents
    documents = [open(file=file, encoding='utf-8', errors="ignore").read() for file in files]
    print("Stanza tagger pre processing all files")
    documents = [stanza_pre_processing(text) for text in documents] #Apply preprocessing
    in_docs = [stanza.Document([], text=d) for d in documents] # Wrap each document with a stanza.Document object
    print("Stanza tagger tagging all files")
    out_docs = nlp(in_docs) # Call the neural pipeline on this list of documents
    for index, doc in enumerate(out_docs):
        #text = open(file=file, encoding='utf-8', errors="ignore").read()
        file = files[index]
        file_name = os.path.basename(file)
        print("Stanza tagger processed:", file)
        #Apply preprocessing
        #text = stanza_pre_processing (text)
        #doc = nlp(text)
        s_list = list()
        for sentence in doc.sentences:
            words = []
            for word in sentence.words:
                words.append(word.text + '_' + word.xpos)
            s_words = " ".join(words)
            s_list.append(s_words)
        s = "\n".join(s_list)
        with open(file=dir_out+file_name, encoding='utf-8', mode='w') as f:
            f.write(s)

def tag_stanford_stanza (dir_in: str, dir_out: str) -> None:
    """Tags text files in dir_in with stanza nlp client and writes to dir_out
    Args:
        dir_in (str): dir with plain text files to be tagged
        dir_out (str): dir to write Stanford Tagger tagged files
    """
    Path(dir_out).mkdir(parents=True, exist_ok=True)   
    #text = open(dir+"corpus\BD-CMT274.txt").read()
    files = glob.glob(dir_in+"*.txt")
    #nlp = stanza.Pipeline('en', processors='tokenize,pos', model_dir=currentdir+"/stanza_resources", download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES, logging_level='WARN', verbose=False, use_gpu=True)
    nlp = stanza.Pipeline('en', processors='tokenize,pos', download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES, logging_level='WARN', verbose=False, use_gpu=True)
    if len(files) > 0:
        if len(files) < 1000:
            process_files_list_chunk_for_stanza(files, nlp, dir_out)
        else:
            n = 1000
            files_list_of_lists = [files[i:i+n] for i in range(0,len(files),n)]
            for index, files_chunk in enumerate(files_list_of_lists):
                print("The corpus contains more than 1000 files which will be divided to chunks of 1000 files. \
                    Processing file chunk number", index+1, "of", len(files_list_of_lists))
                process_files_list_chunk_for_stanza(files_chunk, nlp, dir_out)
    else:
        print("No files to tag.")

        
def process_sentence (words: list, extended: bool = False) -> list:
    """Retunrs words list tagged
    Args:
        words (list): list of words with sentences separated with at least 20 spaces
        extended (bool, optional): extend to add NNP if True. Defaults to False.
    Returns:
        words (list): words list after tagging
    """
    # DICTIONARY LISTS

    have = "have_V|has_V|ve_V|had_V|having_V|hath_|s_VBZ|d_V" # ELF: added s_VBZ, added d_VBD, e.g. "he's got, he's been and he'd been" ELF: Also removed all the apostrophes in Nini's lists because they don't work in combination with \\b in regex as used extensively in this script.
    
    do ="do_V|does_V|did_V|done_V|doing_V|doing_P|done_P" 

    be = "be_V|am_V|is_V|are_V|was_V|were_V|been_V|being_V|s_VBZ|m_V|re_V|been_P" # ELF: removed apostrophes and added "been_P" to account for the verb "be" when tagged as occurrences of passive or perfect forms (PASS and PEAS tags).

    who = "what_|where_|when_|how_|whether_|why_|whoever_|whomever_|whichever_|wherever_|whenever_|whatever_" # ELF: Removed "however" from Nini/Biber's original list.

    wp = "who_|whom_|whose_|which_"

    # ELF: added this list for new WH-question variable:  
    whw = "what_|where_|when_|how_|why_|who_|whom_|whose_|which_" 

    preposition = "about_|against_|amid_|amidst_|among_|amongst_|at_|between_|by_|despite_|during_|except_|for_|from_|in_|into_|minus_|of_|off_|on_|onto_|opposite_|out_|per_|plus_|pro_|than_|through_|throughout_|thru_|toward_|towards_|upon_|versus_|via_|with_|within_|without_" # ELF: removed "besides".

    # ELF: Added this new list but it currently not in use.
    #particles =
    #"about|above|across|ahead|along|apart|around|aside|at|away|back|behind|between|by|down|forward|from|in|into|off|on|out|over|past|through|to|together|under|up|upon|with|without" 

    # ELF: The next three lists of semantic categories of verbs are taken from Biber 1988; however, the current version of the script uses the verb semantic categories from Biber 2006 instead, but the following three lists are still used for some variables, e.g. THATD.
    public = "acknowledge_V|acknowledged_V|acknowledges_V|acknowledging_V|add_V|adds_V|adding_V|added_V|admit_V|admits_V|admitting_V|admitted_V|affirm_V|affirms_V|affirming_V|affirmed_V|agree_V|agrees_V|agreeing_V|agreed_V|allege_V|alleges_V|alleging_V|alleged_V|announce_V|announces_V|announcing_V|announced_V|argue_V|argues_V|arguing_V|argued_V|assert_V|asserts_V|asserting_V|asserted_V|bet_V|bets_V|betting_V|boast_V|boasts_V|boasting_V|boasted_V|certify_V|certifies_V|certifying_V|certified_V|claim_V|claims_V|claiming_V|claimed_V|comment_V|comments_V|commenting_V|commented_V|complain_V|complains_V|complaining_V|complained_V|concede_V|concedes_V|conceding_V|conceded_V|confess_V|confesses_V|confessing_V|confessed_V|confide_V|confides_V|confiding_V|confided_V|confirm_V|confirms_V|confirming_V|confirmed_V|contend_V|contends_V|contending_V|contended_V|convey_V|conveys_V|conveying_V|conveyed_V|declare_V|declares_V|declaring_V|declared_V|deny_V|denies_V|denying_V|denied_V|disclose_V|discloses_V|disclosing_V|disclosed_V|exclaim_V|exclaims_V|exclaiming_V|exclaimed_V|explain_V|explains_V|explaining_V|explained_V|forecast_V|forecasts_V|forecasting_V|forecasted_V|foretell_V|foretells_V|foretelling_V|foretold_V|guarantee_V|guarantees_V|guaranteeing_V|guaranteed_V|hint_V|hints_V|hinting_V|hinted_V|insist_V|insists_V|insisting_V|insisted_V|maintain_V|maintains_V|maintaining_V|maintained_V|mention_V|mentions_V|mentioning_V|mentioned_V|object_V|objects_V|objecting_V|objected_V|predict_V|predicts_V|predicting_V|predicted_V|proclaim_V|proclaims_V|proclaiming_V|proclaimed_V|promise_V|promises_V|promising_V|promised_V|pronounce_V|pronounces_V|pronouncing_V|pronounced_V|prophesy_V|prophesies_V|prophesying_V|prophesied_V|protest_V|protests_V|protesting_V|protested_V|remark_V|remarks_V|remarking_V|remarked_V|repeat_V|repeats_V|repeating_V|repeated_V|reply_V|replies_V|replying_V|replied_V|report_V|reports_V|reporting_V|reported_V|say_V|says_V|saying_V|said_V|state_V|states_V|stating_V|stated_V|submit_V|submits_V|submitting_V|submitted_V|suggest_V|suggests_V|suggesting_V|suggested_V|swear_V|swears_V|swearing_V|swore_V|sworn_V|testify_V|testifies_V|testifying_V|testified_V|vow_V|vows_V|vowing_V|vowed_V|warn_V|warns_V|warning_V|warned_V|write_V|writes_V|writing_V|wrote_V|written_V"
    private = "accept_V|accepts_V|accepting_V|accepted_V|anticipate_V|anticipates_V|anticipating_V|anticipated_V|ascertain_V|ascertains_V|ascertaining_V|ascertained_V|assume_V|assumes_V|assuming_V|assumed_V|believe_V|believes_V|believing_V|believed_V|calculate_V|calculates_V|calculating_V|calculated_V|check_V|checks_V|checking_V|checked_V|conclude_V|concludes_V|concluding_V|concluded_V|conjecture_V|conjectures_V|conjecturing_V|conjectured_V|consider_V|considers_V|considering_V|considered_V|decide_V|decides_V|deciding_V|decided_V|deduce_V|deduces_V|deducing_V|deduced_V|deem_V|deems_V|deeming_V|deemed_V|demonstrate_V|demonstrates_V|demonstrating_V|demonstrated_V|determine_V|determines_V|determining_V|determined_V|discern_V|discerns_V|discerning_V|discerned_V|discover_V|discovers_V|discovering_V|discovered_V|doubt_V|doubts_V|doubting_V|doubted_V|dream_V|dreams_V|dreaming_V|dreamt_V|dreamed_V|ensure_V|ensures_V|ensuring_V|ensured_V|establish_V|establishes_V|establishing_V|established_V|estimate_V|estimates_V|estimating_V|estimated_V|expect_V|expects_V|expecting_V|expected_V|fancy_V|fancies_V|fancying_V|fancied_V|fear_V|fears_V|fearing_V|feared_V|feel_V|feels_V|feeling_V|felt_V|find_V|finds_V|finding_V|found_V|foresee_V|foresees_V|foreseeing_V|foresaw_V|forget_V|forgets_V|forgetting_V|forgot_V|forgotten_V|gather_V|gathers_V|gathering_V|gathered_V|guess_V|guesses_V|guessing_V|guessed_V|hear_V|hears_V|hearing_V|heard_V|hold_V|holds_V|holding_V|held_V|hope_V|hopes_V|hoping_V|hoped_V|imagine_V|imagines_V|imagining_V|imagined_V|imply_V|implies_V|implying_V|implied_V|indicate_V|indicates_V|indicating_V|indicated_V|infer_V|infers_V|inferring_V|inferred_V|insure_V|insures_V|insuring_V|insured_V|judge_V|judges_V|judging_V|judged_V|know_V|knows_V|knowing_V|knew_V|known_V|learn_V|learns_V|learning_V|learnt_V|learned_V|mean_V|means_V|meaning_V|meant_V|note_V|notes_V|noting_V|noted_V|notice_V|notices_V|noticing_V|noticed_V|observe_V|observes_V|observing_V|observed_V|perceive_V|perceives_V|perceiving_V|perceived_V|presume_V|presumes_V|presuming_V|presumed_V|presuppose_V|presupposes_V|presupposing_V|presupposed_V|pretend_V|pretend_V|pretending_V|pretended_V|prove_V|proves_V|proving_V|proved_V|realize_V|realise_V|realising_V|realizing_V|realises_V|realizes_V|realised_V|realized_V|reason_V|reasons_V|reasoning_V|reasoned_V|recall_V|recalls_V|recalling_V|recalled_V|reckon_V|reckons_V|reckoning_V|reckoned_V|recognize_V|recognise_V|recognizes_V|recognises_V|recognizing_V|recognising_V|recognized_V|recognised_V|reflect_V|reflects_V|reflecting_V|reflected_V|remember_V|remembers_V|remembering_V|remembered_V|reveal_V|reveals_V|revealing_V|revealed_V|see_V|sees_V|seeing_V|saw_V|seen_V|sense_V|senses_V|sensing_V|sensed_V|show_V|shows_V|showing_V|showed_V|shown_V|signify_V|signifies_V|signifying_V|signified_V|suppose_V|supposes_V|supposing_V|supposed_V|suspect_V|suspects_V|suspecting_V|suspected_V|think_V|thinks_V|thinking_V|thought_V|understand_V|understands_V|understanding_V|understood_V"
    suasive = "agree_V|agrees_V|agreeing_V|agreed_V|allow_V|allows_V|allowing_V|allowed_V|arrange_V|arranges_V|arranging_V|arranged_V|ask_V|asks_V|asking_V|asked_V|beg_V|begs_V|begging_V|begged_V|command_V|commands_V|commanding_V|commanded_V|concede_V|concedes_V|conceding_V|conceded_V|decide_V|decides_V|deciding_V|decided_V|decree_V|decrees_V|decreeing_V|decreed_V|demand_V|demands_V|demanding_V|demanded_V|desire_V|desires_V|desiring_V|desired_V|determine_V|determines_V|determining_V|determined_V|enjoin_V|enjoins_V|enjoining_V|enjoined_V|ensure_V|ensures_V|ensuring_V|ensured_V|entreat_V|entreats_V|entreating_V|entreated_V|grant_V|grants_V|granting_V|granted_V|insist_V|insists_V|insisting_V|insisted_V|instruct_V|instructs_V|instructing_V|instructed_V|intend_V|intends_V|intending_V|intended_V|move_V|moves_V|moving_V|moved_V|ordain_V|ordains_V|ordaining_V|ordained_V|order_V|orders_V|ordering_V|ordered_V|pledge_V|pledges_V|pledging_V|pledged_V|pray_V|prays_V|praying_V|prayed_V|prefer_V|prefers_V|preferring_V|preferred_V|pronounce_V|pronounces_V|pronouncing_V|pronounced_V|propose_V|proposes_V|proposing_V|proposed_V|recommend_V|recommends_V|recommending_V|recommended_V|request_V|requests_V|requesting_V|requested_V|require_V|requires_V|requiring_V|required_V|resolve_V|resolves_V|resolving_V|resolved_V|rule_V|rules_V|ruling_V|ruled_V|stipulate_V|stipulates_V|stipulating_V|stipulated_V|suggest_V|suggests_V|suggesting_V|suggested_V|urge_V|urges_V|urging_V|urged_V|vote_V|votes_V|voting_V|voted_V"

    # Days of the week ELF: Added to include them in normal noun (NN) count rather than NNP (currently not in use)
    #days = "(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon\.+|Tue\.+|Wed\.+|Thu\.+|Fri\.+|Sat\.+|Sun\.+)"

    # Months ELF: Added to include them in normal noun (NN) count rather than NNP (currently not in use)
    #months = "(January|Jan|February|Feb|March|Mar|April|Apr|May|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)"

    # Stative verbs  
    # ELF: This is a new list which was added on DS's suggestion to count JPRED adjectives more accurately. Predicative adjectives are now identified by exclusion (= adjectives not identified as attributive adjectives) but this dictionary remains useful to disambiguate between PASS and PEAS when the auxiliary is "'s".
    v_stative = "(appear|appears|appeared|feel|feels|feeling|felt|look|looks|looking|looked|become|becomes|became|becoming|get|gets|getting|got|go|goes|going|gone|went|grow|grows|growing|grown|prove|proves|proven|remain|remains|remaining|remained|seem|seems|seemed|shine|shines|shined|shone|smell|smells|smelt|smelled|sound|sounds|sounded|sounding|stay|staying|stayed|stays|taste|tastes|tasted|turn|turns|turning|turned)"

    # Function words
    # EFL: Added in order to calculate a content to function word ratio to capture lexical density
    function_words = "(a|about|above|after|again|ago|ai|all|almost|along|already|also|although|always|am|among|an|and|another|any|anybody|anything|anywhere|are|are|around|as|at|back|be|been|before|being|below|beneath|beside|between|beyond|billion|billionth|both|but|by|can|can|could|cos|cuz|did|do|does|doing|done|down|during|each|eight|eighteen|eighteenth|eighth|eightieth|eighty|either|eleven|eleventh|else|enough|even|ever|every|everybody|everyone|everything|everywhere|except|far|few|fewer|fifteen|fifteenth|fifth|fiftieth|fifty|first|five|for|fortieth|forty|four|fourteen|fourteenth|fourth|from|get|gets|getting|got|had|has|have|having|he|hence|her|here|hers|herself|him|himself|his|hither|how|however|hundred|hundredth|i|if|in|into|is|it|its|itself|just|last|less|many|may|me|might|million|millionth|mine|more|most|much|must|my|myself|near|near|nearby|nearly|neither|never|next|nine|nineteen|nineteenth|ninetieth|ninety|ninth|no|nobody|none|noone|nor|not|nothing|now|nowhere|of|off|often|on|once|one|only|or|other|others|ought|our|ours|ourselves|out|over|quite|rather|round|second|seven|seventeen|seventeenth|seventh|seventieth|seventy|shall|sha|she|should|since|six|sixteen|sixteenth|sixth|sixtieth|sixty|so|some|somebody|someone|something|sometimes|somewhere|soon|still|such|ten|tenth|than|that|that|the|their|theirs|them|themselves|then|thence|there|therefore|these|they|third|thirteen|thirteenth|thirtieth|thirty|this|thither|those|though|thousand|thousandth|three|thrice|through|thus|till|to|today|tomorrow|too|towards|twelfth|twelve|twentieth|twenty|twice|two|under|underneath|unless|until|up|us|very|was|we|were|what|when|whence|where|whereas|which|while|whither|who|whom|whose|why|will|with|within|without|wo|would|yes|yesterday|yet|you|your|yours|yourself|yourselves|'re|'ve|n't|'ll|'twas|'em|y'|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|1|2|3|4|5|6|7|8|9|0)"


    # QUICK CORRECTIONS OF STANFORD POS TAGGER OUTPUT
    for index, x in enumerate(words):
        #skip if space
        if x != " ":
            # Shakir: new feature in MFTE python @mentions
            # Frequent in e-language and now tagged as a new feature category, thus correcting tags such as JJ to NN
            if (re.search(r"^@\S\S+_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_NN NNMention", words[index])

            # Changes the two tags that have a problematic "$" symbol in the Stanford tagset
            if (re.search("PRP\\$", words[index])):                 
                words[index] = re.sub("PRP.", "PRPS", words[index])
                
            if (re.search("WP\\$", words[index])): 
                words[index] = re.sub("WP.", "WPS", words[index]) 

            # ELF: Correction of a few specific symbols identified as adjectives, cardinal numbers, prepositions and foreign words by the Stanford Tagger.
            # These are instead re-tagged as symbols so they don't count as tokens for the TTR and per-word normalisation basis.
            # Removal of all LS (list symbol) tags except those that denote numbers
            if (re.search("<_JJ|>_JJ|\^_FW|>_JJ|§_CD|=_JJ|\*_|\W+_LS|[a-zA-Z]+_LS|\\b@+_|\\b%_|#_NN", words[index])): 
                words[index] = re.sub("_\w+", "_SYM", words[index]) 

            # ELF: Correction of cardinal numbers without spaces and list numbers as numbers rather than LS
            # Removal of the LS (list symbol) tags that denote numbers
            if (re.search("\\b[0-9]+th_|\\b[0-9]+nd_|\\b[0-9]+rd_|[0-9]+_LS", words[index])): 
                words[index] = re.sub("_\w+", "_CD", words[index]) 
                
            # ELF: Correct "innit" and "init" (frequently tagged as a noun by the Stanford Tagger) to pronoun "it" (these are later on also counted as question tags if they are followed by a question mark)
            if (re.search("\\binnit_", words[index])): 
                words[index] = re.sub("_\w+", "_PIT", words[index]) 
            if (re.search("\\binit_", words[index])): 
                words[index] = re.sub("_\w+", "_PIT", words[index]) 	

            # ELF: Correction of the pre- affix frequently tagged as a noun by the POS tagger (stanza)
            if (re.search("\\bpre_", words[index])): 
                words[index] = re.sub("_\w+", "_AFX", words[index]) 

            # ADDITIONAL TAGS FOR INTERNET REGISTERS

            # ELF: New feature for emoji
            # Shakir replaced Elen's regex solution with the emoji module in the Python version of the MFTE
            if (emoji.is_emoji(words[index].split('_')[0])):
                words[index] = re.sub("_\S+", "_EMO", words[index])

            # ELF: New feature for hashtags
            if (re.search("#[a-zA-Z0-9]{3,}_", words[index])):
                words[index] = re.sub("_\w+", "_HST", words[index])

            # ELF: New feature for web links
            # Note that the aim of this regex is *not* to extract all *valid* URLs but rather all strings that were intended to be/look like a URL or a URL-like string!
            # Inspired by: https://mathiasbynens.be/demo/url-regex
            # Designed to favour higher precision over recall.

            if ((re.search("\\b(https?:\/\/www\.|https?:\/\/)?\w+([\-\.\+=&\?]{1}\w+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?", words[index], re.IGNORECASE)) or
                (re.search("<link\/?>", words[index])) or
                (re.search("\\b\w+\.(com|net|co\.uk|au|us|gov|org)\\b", words[index]))):
                words[index] = re.sub("_[\w+\-\.\+=&\/\?]+", "_URL", words[index])

            # BASIC TAG NEEDED FOR MORE COMPLEX TAGS
            # Negation
            if (re.search("\\bnot_|\\bn't_|\\bn’t_|\\bnt_RB", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_XX0", words[index])

    # SLIGHTLY MORE COMPLEX CORRECTIONS OF STANFORD TAGGER OUTPUT

    for j, value in enumerate(words):
        
        if value != " ":

        # Adding the most frequent emoticons to the emoji list
        # Original list: https://repository.upenn.edu/pwpl/vol18/iss2/14/
        # Plus crowdsourced other emoticons from colleagues on Twitter ;-)
        # The regex for this feature aim to favour higher precision over recall in order not to "find" emoticons where they are mostly likely none: e.g., in academic articles with formula.

        # For emoticons parsed as one token by the POS tagger:
        # Extended the regex to include strings including more than one non-alphanumeric character tagged as NFP by the POS tagger.
        # The following were removed because they occur fairly frequently in academic writing ;-RRB-_ and -RRB-:_
            if (re.search(":-RRB-_|:d_|:-LRB-_|:p_|:--RRB-_|:-RSB-_|\\bd:_|:'-LRB-_|:--LRB-_|:-d_|:-LSB-_|-LSB-:_|:-p_|:\/_|:P_|:D_|=-RRB-_|=-LRB-_|:-D_|:-RRB--RRB-_|:O_|:]_|:-LRB--LRB-_|:o_|:-O_|:-o_|;--RRB-_|;-\*|':--RRB--LRB-_|:-B_|8--RRB-_|=\|_|:-\|_|<3_|<\/3_|:P_|;P_|\\bOrz_|\\borz_|\\bXD_|\\bxD_|\\bUwU_|;-\)_|;-\*_|:-\(_|:-\)|;-\(_|:-\(\(_|:-\)\)_|:--\(_|:--\)_|\(ツ|\\b8-\)_|\S\S+_NFP", words[j]) and not re.search("@!_|\.\.\.", words[j])):
                words[j] = re.sub("_\S+", "_EMO", words[j])

            # For emoticons where each character is parsed as an individual token. 
            # N.B.: No longer needed with stanza POS tagging (was necessary with previously used POS tagger)
            # The aim here is to only have one EMO tag per emoticon and, if there are any letters in the emoticon, for the EMO tag to be placed on the letter to overwrite any erroneous NN, FW or LS tags from the Stanford Tagger:
            
            # if ((re.search(":_\W+|;_\W+|=_", words[j]) and re.search("\/_\W+|\\_\W+", words[j+1])) or
            #(re.search(":_|;_|=_", words[j]) and re.search("-LRB-|-RRB-|-RSB-|-LSB-", words[j+1])) or # This line can be used to improve recall when tagging internet registers with lots of emoticons but is not recommended for a broad range of registers since it will cause a serious drop in precision in registers with a lot of punctuation, e.g., academic English.
            # (re.search("\\bd_|\\bp_", words[j], re.IGNORECASE) and re.search(":_", words[j+1])) or
            # (re.search(":_\W+|;_\W+|\\b8_", words[j-2]) and re.search("\\b-_|'_|-LRB-|-RRB-", words[j-1]) and re.search("-LRB-|-RRB-|\\b\_|\\b\/_|\*_", words[j]))):
            #     words[j] = re.sub("_\w+", "_EMO", words[j])
            #     words[j] = re.sub("_(\W+)", "_EMO", words[j])

            # For other emoticons where each character is parsed as an individual token and the letters occur in +1 position.
            # if ((re.search("\\b<_", words[j]) and re.search("\\b3_", words[j+1])) or
            #(re.search(":_|;_|=_", words[j]) and re.search("\\bd_|\\bp_|\\bo_|\\b3_", words[j+1], re.IGNORECASE)) or # # These two lines may be used to improve recall when tagging internet registers with lots of emoticons but is not recommended for a broad range of registers since it will cause a serious drop in precision in registers with a lot of punctuation, e.g., academic English.
            #(re.search("-LRB-|-RRB-|-RSB-|-LSB-", words[j]) and re.search(":_|;_", words[j+1])) or 
            # (re.search(">_", words[j-1]) and re.search(":_", words[j]) and re.search("-LRB-|-RRB-|\\bD_", words[j+1])) or
            # (re.search("\^_", words[j]) and re.search("\^+_", words[j+1])) or
            # (re.search(":_\W+", words[j]) and re.search("\\bo_|\\b-_", words[j+1], re.IGNORECASE) and re.search("-LRB-|-RRB-", words[j+2])) or
            # (re.search("<_", words[j-1]) and re.search("\/_", words[j]) and re.search("\\b3_", words[j+1])) or
            # (re.search(":_\W+|\\b8_|;_\W+|=_", words[j-1]) and re.search("\\b-_|'_|-LRB-|-RRB-", words[j]) and re.search("\\bd_|\\bp_|\\bo_|\\bb_|\\b\|_|\\b\/_", words[j+1], re.IGNORECASE) and not re.search("-RRB-", words[j+2]))):
            #     words[j+1] = re.sub("_\w+", "_EMO", words[j+1])
            #     words[j+1] = re.sub("_(\W+)", "_EMO", words[j+1])

            # Correct double punctuation such as ?! and !? (often tagged by the Stanford Tagger as a noun or foreign word) 
            if (re.search("[\?\!]{2,15}", words[j])):
                words[j] = re.sub("_(\W+)", "_.", words[j])
                words[j] = re.sub("_(\w+)", "_.", words[j])

            # CORRECTION OF "TO" AS PREPOSITION 
            # ELF: Added "to" followed by a punctuation mark, e.g. "What are you up to?"
            if (re.search("\\bto_", words[j], re.IGNORECASE) and re.search("_IN|_CD|_DT|_JJ|_WPS|_NN|_NNP|_PDT|_PRP|_WDT|(\\b(" + wp + "))|_WRB|_\W", words[j+1], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_IN", words[j])

            # ELF: Correction of: "I dunno"
            if (re.search("\\bdu_", words[j], re.IGNORECASE) and re.search("\\bn_", words[j+1]) and re.search("\\bno_", words[j+2])): 
                    words[j] = re.sub("_\w+", "_VPRT", words[j])
                    words[j+1] = re.sub("_\w+", "_XX0", words[j+1])
                    words[j+2] = re.sub("_\w+", "_VB", words[j+2])

            # ELF: Correction of "have" in inverted questions in the present perfect:
            if (re.search("\\bhave_VB", words[j], re.IGNORECASE) and re.search("_PRP", words[j+1]) and re.search("_VBN|_VBD", words[j+2])):
                words[j] = re.sub("_\w+", "_VPRT", words[j])

            # ELF: Correction of falsely tagged "'s" following "there". 
            if (re.search("\\bthere_EX", words[j-1], re.IGNORECASE) and re.search("_POS", words[j])):
                words[j] = re.sub("_\w+", "_VPRT", words[j])

            # ELF: Correction of most problematic spoken language particles
            # ELF: DMA is a new variable. It is important for it to be high up because lots of DMA's are marked as nouns by the Stanford Tagger which messes up other variables further down the line otherwise. 
            # More complex DMAs are tagged further down.
            if (re.search("\\bactually_|\\banyway|\\bdamn_|\\bgoodness_|\\bgosh_|\\byeah_|\\byep_|\\byes_|\\bnope_|\\bright_UH|\\bwhatever_|\\bdamn_RB|\\blol_|\\bIMO_|\\bomg_|\\bwtf_", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_DMA", words[j])

            # ELF: FPUH is a new variable.
            # Aims to tags interjections and filled pauses.
            if (re.search("\\baw+_|\\bow_|\\boh+_|\\beh+_|\\ber+_|\\berm+_|\\bmm+_|\\bum+_|\\b[hu]{2,}_|\\bmhm+|\\bhi+_|\\bhey+_|\\bby+e+_|\\b[ha]{2,}_|\\b[he]{2,}_|\\b[wo]{3,}p?s*_|\\b[oi]{2,}_|\\bouch_|\\bhum+", words[j], re.IGNORECASE)):
                words[j] = re.sub("_(\w+)", "_FPUH", words[j])
            # Also added "hm+" on Peter's suggestion but made sure that this was case sensitive to avoid mistagging Her Majesty ;-)
            if (re.search("\\bhm+|\\bHm+", words[j])):
                words[j] = re.sub("_(\w+)", "_FPUH", words[j])

            # ELF: Added a new "bin" variable for "so" as tagged as a preposition (IN) or adverb (RB) by the Stanford Tagger 
            # because it most often does not seem to be a preposition/conjunct (but rather a filler, amplifier, etc.) and should therefore not be added to the preposition count.

            if (re.search("\\bso_IN|\\bso_RB", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_SO", words[j])

            # Tags quantifiers 
            # ELF: Note that this variable is used to identify several other features. 
            # ELF: added "any", "lots", "loada" and "a lot of" and gave it its own loop because it is now more complex and must be completed before the next set of for-loops. Also added "most" except when later overwritten as an EMPH.
            # ELF: Added "more" and "less" when tagged by the Stanford Tagger as adjectives (JJ.*).
            # ELF: Also added "load(s) of" and "heaps of" on DS's recommendation

            # ELF: Getting rid of the Stanford Tagger predeterminer (PDT) category and now counting all those as quantifiers (QUAN)
            if ((re.search("_PDT", words[j], re.IGNORECASE)) or 
            (re.search("\\ball_|\\bany_|\\bbillions_|\\bboth_|\\bdozens_|\\beach_|\\bevery_|\\bfew_|\\bhalf_|hundreds_|\\bmany_|\\bmillions_|\\bmore_JJ|\\bmuch_|\\bplenty_|\\bseveral_|\\bsome_|\\blots_|\\bloads_|\\bheaps_|\\bless_JJ|\\bloada_|thousands_|\\bwee_|\\bzillions_", words[j], re.IGNORECASE))or
            (re.search("\\bload_|\\bcouple_", words[j], re.IGNORECASE) and re.search("\\bof_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bmost_", words[j], re.IGNORECASE) and re.search("\\bof_|\W+|_N|_J", words[j+1], re.IGNORECASE)) or
            (re.search("\\ba_", words[j-1], re.IGNORECASE) and re.search("\\blot_|\\bbit_|\\blittle_", words[j], re.IGNORECASE)) or # ELF: Added "a lot (of)" and removed NULL tags
            (re.search("\\ba_", words[j-2], re.IGNORECASE) and re.search("\\blot_|\\bbit_", words[j], re.IGNORECASE))): # ELF: Added this line to account for "a little bit", "a whole lot", etc.
                words[j] = re.sub("_\w+", "_QUAN", words[j])
                   
    #---------------------------------------------------
    # COMPLEX TAGS
    for j, value in enumerate(words):

        if value != " ":

        # ELF: DMA is a new variable in the MFTE. 
        #  Here, we tag the remaining pragmatic and discourse markers (see above for simple ones) 
        # The starting point was Stenström's (1994:59) list of "interactional signals and discourse markers" (cited in Aijmer 2002: 2) 
        # --> but it does not include "now" (since it's already a time adverbial), "please" (included in politeness), "quite" or "sort of" (hedges). 
        # Also added: "nope", "I guess", "mind you", "whatever" and "damn" (if not a verb and not already tagged as an emphatic).

            if ((re.search("\\bno_", words[j], re.IGNORECASE) and not re.search("_V", words[j]) and not re.search("_J|_NN|\\bless_", words[j+1])) or # This avoid a conflict with the synthetic negation variable and leaves the "no" in "I dunno" as a present tense verb form and "no" from "no one".
            (re.search("_\W|FPUH_", words[j-1]) and re.search("\\bright_|\\bokay_|\\bok_", words[j], re.IGNORECASE)) or # Right and okay immediately proceeded by a punctuation mark or a filler word
            (not re.search("\\bas_|\\bhow_|\\bvery_|\\breally_|\\bso_|\\bquite_|_V", words[j-1], re.IGNORECASE) and re.search("\\bwell_JJ|\\bwell_RB|\\bwell_NNP|\\bwell_UH", words[j], re.IGNORECASE) and not re.search("_JJ|_RB|-_", words[j+1])) or # Includes all forms of "well" except as a singular noun assuming that the others are mistags of DMA well's by the Stanford Tagger.
            (not re.search("\\bmakes_|\\bmake_|\\bmade_|\\bmaking_|\\bnot|_\\bfor_|\\byou_|\\b(" + be + ")", words[j-1], re.IGNORECASE) and re.search("\\bsure_JJ|\\bsure_RB", words[j], re.IGNORECASE)) or # This excludes MAKE sure, BE sure, not sure, and for sure
            (re.search("\\bof_", words[j-1], re.IGNORECASE) and re.search("\\bcourse_", words[j], re.IGNORECASE)) or
            (re.search("\\ball_", words[j-1], re.IGNORECASE) and re.search("\\bright_", words[j], re.IGNORECASE)) or
            (re.search("\\bmind_", words[j], re.IGNORECASE) and re.search("\\byou_", words[j+1], re.IGNORECASE))): 

                words[j] = re.sub("_\w+", "_DMA", words[j])

        #--------------------------------------------------

            # Tags attribute adjectives (JJAT) (see additional loop further down the line for additional JJAT cases that rely on these JJAT tags)

            if ((re.search("_JJ", words[j]) and re.search("_JJ|_NN|_CD", words[j+1])) or
            (re.search("_DT", words[j-1]) and re.search("_JJ", words[j]))):
                words[j] = re.sub("_\w+", "_JJAT", words[j])

            #----------------------------------------------------    
            
            # Manually add okay as a predicative adjective (JJPR) because "okay" and "ok" are often tagged as foreign words by the Stanford Tagger. All other predicative adjectives are tagged at the very end.

            if (re.search("\\b(" + be + ")", words[j-1], re.IGNORECASE) and re.search("\\bok_|okay_", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_JJPR", words[j])

            #---------------------------------------------------

            # Tags elaborating conjunctions (ELAB)
            # ELF: This is a new variable.

            # ELF: added the exception that "that" should not be a determiner. Also added "in that" and "to the extent that" on DS's advice.  

            if ((re.search("\\bsuch_", words[j-1], re.IGNORECASE) and re.search("\\bthat_", words[j]) and not re.search("_DT", words[j])) or
            (re.search("\\bsuch_|\\binasmuch__|\\bforasmuch_|\\binsofar_|\\binsomuch", words[j-1], re.IGNORECASE) and re.search("\\bas_", words[j])) or
            (re.search("\\bso_", words[j-2], re.IGNORECASE) and re.search("\\blong_", words[j-1]) and re.search("\\bas_", words[j])) or
            (re.search("\\bin_IN", words[j-1], re.IGNORECASE) and re.search("\\bthat_", words[j]) and not re.search("_DT", words[j])) or
            (re.search("\\bto_", words[j-3], re.IGNORECASE) and re.search("\\bthe_", words[j-2]) and re.search("\\bextent_", words[j-1]) and re.search("\\bthat_", words[j])) or
            (re.search("\\bin_", words[j-1], re.IGNORECASE) and re.search("\\bparticular_|\\bconclusion_|\\bsum_|\\bsummary_|\\bfact_|\\bbrief_", words[j], re.IGNORECASE)) or
            (re.search("\\bto_", words[j-1], re.IGNORECASE) and re.search("\\bsummarise_|\\bsummarize_", words[j], re.IGNORECASE) and re.search(",_", words[j])) or
            (re.search("\\bin_", words[j-1], re.IGNORECASE) and re.search("\\bshort_", words[j], re.IGNORECASE) and re.search(",_", words[j])) or
            (re.search("\\bfor_", words[j-1], re.IGNORECASE) and re.search("\\bexample_|\\binstance_", words[j], re.IGNORECASE)) or
            (re.search("\\bsimilarly_|\\baccordingly_", words[j], re.IGNORECASE) and re.search(",_", words[j+1])) or
            (re.search("\\bin_", words[j-2], re.IGNORECASE) and re.search("\\bany_", words[j-1], re.IGNORECASE) and re.search("\\bevent_|\\bcase_", words[j], re.IGNORECASE)) or
            (re.search("\\bin_", words[j-2], re.IGNORECASE) and re.search("\\bother_", words[j-1], re.IGNORECASE) and re.search("\\bwords_", words[j]))):
                words[j] = re.sub("_(\w+)", "_\\1 ELAB", words[j])

            if (re.search("\\beg_|\\be\.g\._|etc\.?_|\\bi\.e\._|\\bcf\.?_|\\blikewise_|\\bnamely_|\\bviz\.?_", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_ELAB", words[j])

            #---------------------------------------------------

            # Tags coordinating conjunctions (CC)
            # ELF: This is a new variable.
            # ELF: added as well (as), as well, in fact, accordingly, thereby, also, by contrast, besides, further_RB, in comparison, instead (not followed by "of").
            if ((re.search("\\bwhile_IN|\\bwhile_RB|\\bwhilst_|\\bwhereupon_|\\bwhereas_|\\bwhereby_|\\bthereby_|\\balso_|\\bbesides_|\\bfurther_RB|\\binstead_|\\bmoreover_|\\bfurthermore_|\\badditionally_|\\bhowever_|\\binstead_|\\bibid\._|\\bibid_|\\bconversly_", words[j], re.IGNORECASE)) or 
            (re.search("\\binasmuch__|\\bforasmuch_|\\binsofar_|\\binsomuch", words[j], re.IGNORECASE) and re.search("\\bas_", words[j+1], re.IGNORECASE)) or
            (re.search("_\W", words[j-1], re.IGNORECASE) and re.search("\\bhowever_", words[j], re.IGNORECASE)) or
            (re.search("_\W", words[j+1], re.IGNORECASE) and re.search("\\bhowever_", words[j], re.IGNORECASE)) or
            (re.search("\\bor_", words[j-1], re.IGNORECASE) and re.search("\\brather_", words[j], re.IGNORECASE)) or
            (not re.search("\\bleast_", words[j-1], re.IGNORECASE) and re.search("\\bas_", words[j], re.IGNORECASE) and re.search("\\bwell_", words[j+1], re.IGNORECASE)) or # Excludes "at least as well" but includes "as well as"
            (re.search("_\W", words[j-1]) and re.search("\\belse_|\\baltogether_|\\brather_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_CC", words[j])


            if ((re.search("\\bby_", words[j-1], re.IGNORECASE) and re.search("\\bcontrast_|\\bcomparison_", words[j], re.IGNORECASE)) or
            (re.search("\\bin_", words[j-1], re.IGNORECASE) and re.search("\\bcomparison_|\\bcontrast_|\\baddition_", words[j], re.IGNORECASE)) or
            (re.search("\\bon_", words[j-2], re.IGNORECASE) and re.search("\\bthe_", words[j-1]) and re.search("\\bcontrary_", words[j], re.IGNORECASE)) or
            (re.search("\\bon_", words[j-3], re.IGNORECASE) and re.search("\\bthe_", words[j-2]) and re.search("\\bone_|\\bother_", words[j-1], re.IGNORECASE) and re.search("\\bhand_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_\\1 CC", words[j])
        
            #---------------------------------------------------

            # Tags causal conjunctions     
            # ELF added: cos, cus, coz, cuz and 'cause (a form spotted in one textbook of the TEC!) plus all the complex forms below.
            if ((re.search("\\bbecause_|\\bcos_|\\bcos\._|\\bcus_|\\bcuz_|\\bcoz_|\\b'cause_", words[j], re.IGNORECASE)) or
            (re.search("\\bconsequently_|\\bhence_|\\btherefore_", words[j], re.IGNORECASE)) or
            (re.search("\\bthanks_", words[j], re.IGNORECASE) and re.search("\\bto_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bthus_", words[j], re.IGNORECASE) and not re.search("\\bfar_", words[j+1], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_CUZ", words[j])

            if ((re.search("\\bin_", words[j-1], re.IGNORECASE) and re.search("\\bconsequence_", words[j], re.IGNORECASE)) or
            (re.search("\\bsuch_|\\bso_", words[j-1], re.IGNORECASE) and re.search("\\bthat_", words[j]) and not re.search("_DT", words[j])) or
            (re.search("\\bas_", words[j-2], re.IGNORECASE) and re.search("\\ba_", words[j-1], re.IGNORECASE) and re.search("\\bresult_|\\bconsequence_", words[j], re.IGNORECASE)) or
            (re.search("\\bon_", words[j-2], re.IGNORECASE) and re.search("\\baccount_", words[j-1], re.IGNORECASE) and re.search("\\bof_", words[j], re.IGNORECASE)) or
            (re.search("\\bfor_", words[j-2], re.IGNORECASE) and re.search("\\bthat_|\\bthis_", words[j-1], re.IGNORECASE) and re.search("\\bpurpose_", words[j], re.IGNORECASE)) or
            (re.search("\\bto_", words[j-2], re.IGNORECASE) and re.search("\\bthat_|\\bthis_", words[j-1], re.IGNORECASE) and re.search("\\bend_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_\\1 CUZ", words[j])

            #---------------------------------------------------

            # Tags conditional conjunctions
            # ELF: added "lest" on DS's suggestion. Added "whether" on PU's suggestion.

            if (re.search("\\bif_|\\bunless_|\\blest_|\\botherwise_|\\bwhether_", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_COND", words[j])

            if ((re.search("\\bas_", words[j-2], re.IGNORECASE) and re.search("\\blong_", words[j-1]) and re.search("\\bas_", words[j])) or
            (re.search("\\bin_", words[j-2], re.IGNORECASE) and re.search("\\bthat_", words[j-1]) and re.search("\\bcase_", words[j]))):
                words[j] = re.sub("_(\w+)", "_\\1 COND", words[j])

            #---------------------------------------------------

            # Tags emphatics 
            # ELF: added "such an" and ensured that the indefinite articles in "such a/an" are not tagged as NULL as was the case in Nini's script. Removed "more".
            # Added: so many, so much, so little, so + VERB, damn + ADJ, least, bloody, fuck, fucking, damn, super and dead + ADJ.
            # Added a differentiation between "most" as as QUAN ("most of") and EMPH.
            # Improved the accuracy of DO + verb by specifying a base form (_VB) so as to avoid: "Did they do_EMPH stuffed_VBN crust?".
            if ((re.search("\\bmost_DT", words[j], re.IGNORECASE)) or
            (re.search("\\bmost_", words[j], re.IGNORECASE) and re.search("_J|_VBN|_VBG", words[j+1])) or
            (re.search("\\breal_|\\bdead_|\\bdamn_|\\bfuck|\\bshit|\\bsuper_", words[j], re.IGNORECASE) and re.search("_J|_RB", words[j+1])) or
            (re.search("\\bjust_|\\breally_|\\bbloody_|\\bpretty_|\\bmore_", words[j], re.IGNORECASE) and re.search("_J|_RB|_V", words[j+1], re.IGNORECASE)) or
            (re.search("\\bso_", words[j], re.IGNORECASE) and re.search("_J|\\bmany_|\\bmuch_|\\blittle_|_RB", words[j+1], re.IGNORECASE) and not re.search("\\bfar_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bfar_", words[j], re.IGNORECASE) and re.search("_J|_RB", words[j+1]) and not re.search("\\bso_|\\bthus_", words[j-1], re.IGNORECASE)) or
            (not re.search("\\bof_", words[j-1], re.IGNORECASE) and re.search("\\bsuch_", words[j], re.IGNORECASE) and re.search("\\ba_|\\ban_", words[j+1], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_EMPH", words[j])

            if ((re.search("\\bloads_", words[j], re.IGNORECASE) and not re.search("\\bof_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bfor_", words[j], re.IGNORECASE) and re.search("\\bsure_", words[j+1], re.IGNORECASE))): 
                words[j] = re.sub("_(\w+)", "_\\1 EMPH", words[j])

            if (re.search("\\b(" + do + ")", words[j], re.IGNORECASE) and re.search("_VB\\b", words[j+1])):                
                words[j] = re.sub("_(\w+)", "_\\1 DOAUX EMPH", words[j])
            #---------------------------------------------------

            # Feature from the MAT/Biber Tagger.
            # Tags phrasal coordination with "and", "or" and "nor". 
            # ELF: Not currently in use due to low precision and recall (see perl tagger performance evaluation).
            #if ((re.search("\\band_|\\bor_|&_|\\bnor_", words[j], re.IGNORECASE)) and
            # ((re.search("_RB", words[j-1]) and re.search("_RB", words[j+1])) or
            #(re.search("_J", words[j-1]) and re.search("_J", words[j+1])) or
            #(re.search("_V", words[j-1]) and re.search("_V", words[j+1])) or
            #(re.search("_CD", words[j-1]) and re.search("_CD", words[j+1])) or
            #(re.search("_NN", words[j-1]) and re.search("_NN|whatever_|_DT", words[j+1])))):
            #   words[j] = re.sub("_\w+", "_PHC", words[j])

            #---------------------------------------------------

            # Tags auxiliary DO ELF: I added this variable which replaces the MAT's/Biber tagger's "pro-verb" DO variable. 
            # Later on, all DO verbs not tagged as DOAUX are tagged as ACT.
            if (re.search("\\bdo_V|\\bdoes_V|\\bdid_V", words[j], re.IGNORECASE) and not re.search("to_TO", words[j-1])): # This excludes DO + VB\\b which have already been tagged as emphatics (DO_EMPH) and "to do" constructions
                if ((re.search("_VB\\b", words[j+2])) or # did you hurt yourself? Didn't look? 
                (re.search("_VB\\b", words[j+3])) or # didn't it hurt?
                (re.search("_\W", words[j+1])) or # You did?
                (re.search("\\bI_|\\byou_|\\bhe_|\\bshe_|\\bit_|\\bwe_|\\bthey_|_XX0", words[j+1], re.IGNORECASE) and re.search("_\.|_VB\\b", words[j+2])) or # ELF: Added to include question tags such as: "do you?"" or "He didn't!""
                (re.search("_XX0", words[j+1]) and re.search("\\bI_|\\byou_|\\bhe_|\\bshe_|\\bit_|\\bwe_|\\bthey_|_VB\\b", words[j+2], re.IGNORECASE)) or # Allows for question tags such as: didn't you? as well as negated forms such as: did not like
                (re.search("\\bI_|\\byou_|\\bhe_|\\bshe_|\\bit_|\\bwe_|\\bthey_", words[j+1], re.IGNORECASE) and re.search("\\?_\\.", words[j+3])) or # ELF: Added to include question tags such as: did you not? did you really?
                (re.search("(\\b(" + wp + "))|(\\b" + who + ")|(\\b" + whw + ")", words[j-1], re.IGNORECASE))):
                    words[j] = re.sub("_(\w+)", "_\\1 DOAUX", words[j])
                
        #---------------------------------------------------

        # ELF: Regex for question tags. New variable in MFTE.

            if ((not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("_MD|\\bdid_|\\bhad_", words[j-3], re.IGNORECASE) and re.search("_XX0", words[j-2]) and re.search("_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("\\?_\\.", words[j])) or # couldn't he?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("_MD|\\bdid_|\\bhad_|\\bdo_", words[j-2], re.IGNORECASE) and re.search("_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_|\\byou_", words[j-1]) and re.search("\\?_\\.", words[j])) or # did they?
            (not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("\\bis_|\\bdoes_|\\bwas|\\bhas|\\bdo_", words[j-3], re.IGNORECASE) and re.search("_XX0", words[j-2]) and re.search("\\bit_|\\bshe_|\\bhe_|\\bthey_", words[j-1], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # isn't it?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("\\bis_|\\bdoes_|\\bwas|\\bhas_", words[j-2], re.IGNORECASE) and re.search("\\bit_|\\bshe_|\\bhe_", words[j-1], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # has she?
            (not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("\\bdo_|\\bwere_|\\bare_|\\bhave_", words[j-3], re.IGNORECASE) and re.search("_XX0", words[j-2]) and re.search("\\byou_|\\bwe_|\\bthey_", words[j-1], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # haven't you?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("\\bdo_|\\bwere_|\\bare_|\\bhave_", words[j-2], re.IGNORECASE) and re.search("\\byou_|\\bwe_|\\bthey_", words[j-1], re.IGNORECASE) and re.search("\\?_\\.", words[j])) or # were you?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("_MD|\\bdid_|\\bhad_|\\bdo_", words[j-1], re.IGNORECASE) and re.search("_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_|\\byou_", words[j-2]) and re.search("\\?_\\.", words[j])) or # they did?
            (not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("\\bis_|\\bdoes_|\\bwas|\\bhas|\\bdo_", words[j-2], re.IGNORECASE) and re.search("_XX0", words[j-1]) and re.search("\\bit_|\\bshe_|\\bhe_|\\bthey_", words[j-3], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # it isn't?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("\\bis_|\\bdoes_|\\bwas|\\bhas_", words[j-1], re.IGNORECASE) and re.search("\\bit_|\\bshe_|\\bhe_", words[j-2], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # she has?
            (not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("\\bdo_|\\bwere_|\\bare_|\\bhave_", words[j-2], re.IGNORECASE) and re.search("_XX0", words[j-1]) and re.search("\\byou_|\\bwe_|\\bthey_", words[j-3], re.IGNORECASE) and re.search("\\?_\\.", words[j]))  or # you haven't?
            (not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("\\bdo_|\\bwere_|\\bare_|\\bhave_", words[j-1], re.IGNORECASE) and re.search("\\byou_|\\bwe_|\\bthey_", words[j-2], re.IGNORECASE) and re.search("\\?_\\.", words[j])) or # you were?
            (not re.search("\\b(" + whw + ")", words[j-2], re.IGNORECASE) and re.search("\\binnit_|\\binit_", words[j-1]) and re.search("\\?_\\.", words[j]))): # innit? init?

                words[j] = re.sub("_(\W+)", "_\\1 QUTAG", words[j])


            #---------------------------------------------------    
            # Tags yes/no inverted questions (YNQU)
            # ELF: New variable and new operationalistion for python version with tags added to the question marks as opposed to the verbs
            # Note that, at this stage in the script, DT still includes demonstrative pronouns which is good. 
            # Also _P, at this stage, only includes PRP, and PPS (i.e., not yet any of the new verb variables which should not be captured here)
            
            if ((not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-3], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-2], re.IGNORECASE) and re.search("_P|_NN|_DT|_CD", words[j-1]) and re.search("\\?_\\.$", words[j])) or  # Are they there? It is him?
            (not re.search("\\b(" + whw + ")", words[j-6], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-4], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-3], re.IGNORECASE) and re.search("_P|_NN|_DT|_CD", words[j-2]) and re.search("\\?_\\.$", words[j])) or # Can you tell him?
            (not re.search("\\b(" + whw + ")", words[j-7], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-6], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-5], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-4], re.IGNORECASE) and re.search("_P|_NN|_DT_CD", words[j-3]) and re.search("\\?_\\.$", words[j])) or # Did her boss know that?
            (not re.search("\\b(" + whw + ")", words[j-8], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-7], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-6], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-5], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-4]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-9], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-8], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-7], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-6], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-5]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-10], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-9], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-8], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-7], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-6]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-11], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-10], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-9], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-8], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-7]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-12], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-11], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-10], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-9], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-8]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-13], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-12], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-11], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-10], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-9]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-14], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-13], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-12], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-11], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-10]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-15], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-14], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-13], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-12], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-11]) and re.search("\\?_\\.$", words[j])) or
            (not re.search("\\b(" + whw + ")", words[j-16], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-15], re.IGNORECASE) and not re.search("\\b(" + whw + ")", words[j-14], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j-13], re.IGNORECASE) and re.search("_P|_NN|_DT|_XX0|_CD", words[j-12]) and re.search("\\?_\\.$", words[j]))):
                words[j] = re.sub("_(\W+)", "_\\1 YNQU", words[j])

            #---------------------------------------------------

            # Tags passives 
            # ELF: merged Biber's BYPA and PASS categories together into one and changed the original coding procedure on its head: this script now tags the past participles rather than the verb BE. It also allows for mistagging of -ed past participle forms as VBD by the Stanford Tagger.
            # ELF: I am including most "'s_VBZ" as a possible form of the verb BE here but later on overriding many instances as part of the PEAS variable.    

            if (re.search("_VBN|ed_VBD|en_VBD", words[j])): # Also accounts for past participle forms ending in "ed" and "en" mistagged as past tense forms (VBD) by the Stanford Tagger

                if ((re.search("\\b(" + be + ")", words[j-1], re.IGNORECASE)) or # is eaten 
                #(re.search("s_VBZ", words[j-1], re.IGNORECASE) and re.search("\\bby_", words[j+1])) or # This line enables the passive to be preferred over present perfect if immediately followed by a "by"
                (re.search("_RB|_XX0|_CC", words[j-1]) and re.search("\\b(" + be + ")", words[j-2], re.IGNORECASE)) or # isn't eaten 
                (re.search("_RB|_XX0|_CC", words[j-1]) and re.search("_RB|_XX0", words[j-2]) and re.search("\\b(" + be + ")", words[j-3], re.IGNORECASE) and not re.search("\\bs_VBZ", words[j-3])) or # isn't really eaten
                (re.search("_NN|_PRP|_CC", words[j-1]) and re.search("\\b(" + be + ")", words[j-2], re.IGNORECASE))or # is it eaten
                (re.search("_RB|_XX0|_CC", words[j-1]) and re.search("_NN|_PRP", words[j-2]) and re.search("\\b(" + be + ")", words[j-3], re.IGNORECASE) and not re.search("\\bs_VBZ", words[j-3]))): # was she not failed?
                    words[j] = re.sub("_\w+", "_PASS", words[j])

            # ELF: Added a new variable for GET-passives
            if (re.search("_VBD|_VBN", words[j])):
                if ((re.search("\\bget_V|\\bgets_V|\\bgot_V|\\bgetting_V", words[j-1], re.IGNORECASE)) or
                (re.search("_NN|_PRP", words[j-1]) and re.search("\\bget_V|\\bgets_V|\\bgot_V|\\bgetting_V", words[j-2], re.IGNORECASE)) or # She got it cleaned
                (re.search("_NN", words[j-1]) and re.search("_DT|_NN", words[j-2]) and re.search("\\bget_V|\\bgets_V|\\bgot_V|\\bgetting_V", words[j-3], re.IGNORECASE))): # She got the car cleaned
                    words[j] = re.sub("_\w+", "_PGET", words[j])

            #---------------------------------------------------

            # ELF: Added the new variable GOING TO, which allows for one intervening word between TO and the infinitive
            # Shakir: Added case insensitive flags for "going" and "gon"
            
            if ((re.search("\\bgoing_VBG", words[j], re.IGNORECASE) and re.search("\\bto_TO", words[j+1]) and re.search("\_VB", words[j+2])) or
            (re.search("\\bgoing_VBG", words[j], re.IGNORECASE) and re.search("\\bto_TO", words[j+1]) and re.search("\_VB", words[j+3])) or
            (re.search("\\bgon_VBG", words[j], re.IGNORECASE) and re.search("\\bna_", words[j+1]) and re.search("\_VB", words[j+2])) or
            (re.search("\\bgon_VBG", words[j], re.IGNORECASE) and re.search("\\bna_", words[j+1]) and re.search("\_VB", words[j+3]))):
                words[j] = re.sub("_\w+", "_GTO", words[j])

            #----------------------------------------------------

            # Tags synthetic negation 
            # ELF: I'm merging this category with Biber's original analytic negation category (XX0) so I've had to move it further down in the script so it doesn't interfere with other complex tags
            if ((re.search("\\bno_", words[j], re.IGNORECASE) and re.search("_J|_NN|\\blonger_|\\bmore_|\\bneed_|\\bdoubt_|\\bpoint_|\\breason_|\\bsuch_|\\bproblem_|\\bmatter_|\\bmeans_", words[j+1])) or
            (re.search("\\bneither_", words[j], re.IGNORECASE)) or
            (re.search("\\bnor_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_XX0", words[j])

            # Added a loop to tag "no one" and "each other" as a QUPR
            if ((re.search("\\bno_", words[j], re.IGNORECASE) and re.search("\\bone_", words[j+1])) or
            (re.search("\\beach_", words[j-1], re.IGNORECASE) and re.search("\\bother_", words[j]))):
                words[j+1] = re.sub("_(\w+)", "_QUPR", words[j+1])

            #---------------------------------------------------

            # Tags split infinitives
            # ELF: merged this variable with split auxiliaries due to very low counts. Also removed "_AMPLIF|_DOWNTON" from these lists which Nini had but which made no sense because AMP and DWNT are a) tagged with shorter acronyms and b) this happens in future loops so RB does the job here. However, RB does not suffice for "n't" and not so I added _XX0 to the regex.
            if ((re.search("\\bto_", words[j], re.IGNORECASE) and re.search("_RB|\\bjust_|\\breally_|\\bmost_|\\bmore_|_XX0", words[j+1], re.IGNORECASE) and re.search("_V", words[j+2])) or
            (re.search("\\bto_", words[j], re.IGNORECASE) and re.search("_RB|\\bjust_|\\breally_|\\bmost_|\\bmore_|_XX0", words[j+1], re.IGNORECASE) and re.search("_RB|_XX0", words[j+2]) and re.search("_V", words[j+3])) or

            # Tags split auxiliaries - ELF: merged this variable with split infinitives due to very low counts. 
            # ELF: Also changed all forms of DO to auxiliary DOs only 
            (re.search("_MD|DOAUX|(\\b(" + have + "))|(\\b(" + be + "))", words[j], re.IGNORECASE) and re.search("_RB|\\bjust_|\\breally_|\\bmost_|\\bmore_", words[j+1], re.IGNORECASE) and re.search("_V", words[j+2])) or
            (re.search("_MD|DOAUX|(\\b(" + have + "))|(\\b(" + be + "))", words[j], re.IGNORECASE) and re.search("_RB|\\bjust_|\\breally_|\\bmost_|\\bmore_|_XX0", words[j+1], re.IGNORECASE) and re.search("_RB|_XX0", words[j+2]) and re.search("_V", words[j+3]))):
                words[j] = re.sub("_(\w+)", "_\\1 SPLIT", words[j])

            #---------------------------------------------------

            # ELF: Attempted to add an alternative stranded "prepositions/particles"
            # This is currently not in use because it's too inaccurate.
            #if (re.search("\\b(" + particles + ")_IN|\\b(" + particles + ")_RP|\\b(" + particles + ")_RB|to_TO", words[j], re.IGNORECASE) and re.search("_\W", words[j+1])):
            # words[j] = re.sub("_(\w+)", "_\\1 [STPR]", words[j])

            # Tags stranded prepositions
            # ELF: completely changed Nini's regex because it relied on PIN which is no longer a variable in use in the MFTE. 
            if (re.search("\\b(" + preposition + ")|\\bto_TO", words[j], re.IGNORECASE) and not re.search("_R", words[j]) and re.search("_\.", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 STPR", words[j])

            #---------------------------------------------------
            # Tags imperatives (in a rather crude way). 
            # ELF: This is a new variable in the MFTE.
            if ((re.search("_\\.|:|-_NFP|_EMO|_FW|_SYM|_HST| $|\\bplease_|\\bPlease_", words[j-1]) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX|\\b(" + be + ")", words[j], re.IGNORECASE) and not re.search("\\bI_|\\byou_|\\bwe_|\\bthey_|_NNP", words[j+1], re.IGNORECASE)) or # E.g., "This is a task. Do it." # Added _SYM and _FW because imperatives often start with bullet points which are not always recognised as such. Also added _EMO for texts that use emoji/emoticons instead of punctuation.
            #(re.search("_\W|_EMO|_FW|_SYM", words[j-2])  and not re.search("_,", words[j-2]) and not re.search("_MD", words[j-1]) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX|\\b(" + be + ")", words[j], re.IGNORECASE) and not re.search("\\bI_|\\byou_|\\bwe_|\\bthey_|\\b_NNP", words[j+1], re.IGNORECASE)) or # Allows for one intervening token between end of previous sentence and imperative verb, e.g., "Just do it!". This line is not recommended for the Spoken BNC2014 and any texts with not particularly good punctuation.
            (re.search("_\\.|:|-_NFP|_EMO|_FW|_SYM|_HST| $", words[j-2]) and re.search("_RB|_CC|_DMA", words[j-1]) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX|\\b(" + be + ")", words[j], re.IGNORECASE) and not re.search("\\bI_|\\byou_|\\bwe_|\\bthey_|_NNP", words[j+1])) or # "Listen carefully. Then fill the gaps."
            (re.search("_\\.|:|-_NFP|_EMO|_FW|_SYM|_HST| $|\\bplease_|\\bPlease_", words[j-1]) and re.search("\\bpractise_|\\bmake_|\\bcomplete", words[j], re.IGNORECASE)) or
            #(re.search("\\bPractise_|\\bMake_|\\bComplete_|\\bMatch_|\\bRead_|\\bChoose_|\\bWrite_|\\bListen_|\\bDraw_|\\bExplain_|\\bThink_|\\bCheck_|\\bDiscuss_", words[j])) or # Most frequent imperatives that start sentences in the Textbook English Corpus (TEC) (except "Answer" since it is genuinely also frequently used as a noun)
            (re.search("_\\.|:|-_NFP|_EMO|_FW|_SYM|_HST| $|\\bplease_|\\bPlease_", words[j-1]) and re.search("\\bdo_", words[j], re.IGNORECASE) and re.search("_XX0", words[j+1]) and re.search("_VB\\b", words[j+2], re.IGNORECASE)) or # Do not write. Don't listen.
            (re.search("_\\.|:|-_NFP|_EMO|_FW|_SYM|_HST| $", words[j-2]) and re.search("_RB|_CC|_DMA", words[j-1]) and re.search("\\bdo_", words[j], re.IGNORECASE) and re.search("_XX0", words[j+1]) and re.search("_VB\\b", words[j+2], re.IGNORECASE))): # Do not write. Don't listen.
            #(re.search("\\bwork_", words[j], re.IGNORECASE) and re.search("\\bin_", words[j+1], re.IGNORECASE) and re.search("\\bpairs_", words[j+2], re.IGNORECASE))): # Work in pairs because it occurs 700+ times in the Textbook English Corpus (TEC) and "work" is always incorrectly tagged as a noun there.
                words[j] = re.sub("_\w+", "_VIMP", words[j]) 
            
            if ((re.search("_VIMP", words[j-2]) and re.search("\\band_|\\bor_|,_|&_", words[j-1], re.IGNORECASE) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX", words[j], re.IGNORECASE)) or
            (re.search("_VIMP", words[j-3]) and re.search("\\band_|\\bor_|,_|&_", words[j-1], re.IGNORECASE) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX", words[j], re.IGNORECASE)) or
            (re.search("_VIMP", words[j-4]) and re.search("\\band_|\\bor_|,_|&_", words[j-1], re.IGNORECASE) and re.search("_VB\\b", words[j]) and not re.search("\\bplease_|\\bthank_| DOAUX", words[j], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_VIMP", words[j]) # This accounts for, e.g., "read (carefully/the text) and listen"

            #---------------------------------------------------

            # Tags 'that' adjective complements. 
            # ELF: added the _IN tag onto the "that" to improve accuracy but currently not in use because it still proves too error-prone.
            #if (re.search("_J", words[j-1]) and re.search("\\bthat_IN", words[j], re.IGNORECASE)):
            # words[j] = re.sub("_\w+", "_THAC", words[j])

            # ELF: tags other adjective complements. It's important that WHQU comes afterwards.
            # ELF: also currently not in use because of the high percentage of tagging errors.
            #if (re.search("_J", words[j-1]) and re.search("\\bwho_|\\bwhat_WP|\\bwhere_|\\bwhy_|\\bhow_|\\bwhich_", words[j], re.IGNORECASE)):
            # words[j] = re.sub("_\w+", "_WHAC", words[j])

            #---------------------------------------------------

            # ELF: Removed Biber's more detailed and, without manual adjustments, highly unreliable variables WHSUB, WHOBJ, THSUB, THVC, and TOBJ
            # Replaced them with much simpler variables (see Shakir's extended tagset for more finer-grained features). 
            # It should be noted, however, that these variables rely much more on the Stanford Tagger which is far from perfect depending on the type of texts to be tagged. 
            # Thorough manual checks are therefore still highly recommended before using the counts of these variables!
            
            # That-subordinate clauses other than relatives according to the Stanford Tagger  
            if (re.search("\\bthat_IN", words[j], re.IGNORECASE) and not re.search("_\W", words[j+1])):
                words[j] = re.sub("_\w+", "_THSC", words[j])

            # That-relative clauses according to the Stanford Tagger  
            if (re.search("\\bthat_WDT", words[j], re.IGNORECASE) and not re.search("_\W", words[j+1])):
                words[j] = re.sub("_\w+", "_THRC", words[j])

            # Subordinate clauses with WH-words. 
            # ELF: New variable in the MFTE.
            if (re.search("\\b(" + wp + ")|\\b(" + who + ")", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_WHSC", words[j])

            #---------------------------------------------------
            # Tags hedges 
            # ELF: added "kinda" and "sorta" and corrected the "sort of" and "kind of" lines in Nini's original script which had the word-2 part negated.
            # Also added apparently, conceivably, perhaps, possibly, presumably, probably, roughly and somewhat.
            if ((re.search("\\bmaybe_|apparently_|conceivably_|perhaps_|\\bpossibly_|presumably_|\\bprobably_|\\broughly_|somewhat_|\\bpredictably_", words[j], re.IGNORECASE)) or
            (re.search("\\baround_|\\babout_", words[j], re.IGNORECASE) and re.search("_CD|_QUAN|_$|_SYM", words[j+1], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_HDG", words[j])

            if ((re.search("\\bat_", words[j-1], re.IGNORECASE) and re.search("\\babout_", words[j], re.IGNORECASE)) or
            (re.search("\\bsomething_", words[j-1], re.IGNORECASE) and re.search("\\blike_", words[j], re.IGNORECASE)) or
            (not re.search("_DT|_QUAN|_CD|_J|_PRP|(\\b(" + who + "))", words[j-2], re.IGNORECASE) and re.search("\\bsort_", words[j-1], re.IGNORECASE) and re.search("\\bof_", words[j], re.IGNORECASE)) or
            (not re.search("_DT|_QUAN|_CD|_J|_PRP|(\\b(" + who + "))", words[j-2], re.IGNORECASE) and re.search("\\bkind_NN", words[j-1], re.IGNORECASE) and re.search("\\bof_", words[j], re.IGNORECASE)) or
            (not re.search("_DT|_QUAN|_CD|_J|_PRP|(\\b(" + who + "))", words[j-1], re.IGNORECASE) and re.search("\\bkinda_|\\bsorta_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_\\1 HDG", words[j])

            if (re.search("\\bmore_", words[j-2], re.IGNORECASE) and re.search("\\bor_", words[j-1], re.IGNORECASE) and re.search("\\bless_", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_QUAN HDG", words[j])
                words[j-2] = re.sub("\w+", "_QUAN", words[j-2])

            #Shakir: multiword likelihood adverbs to HDG
            if (re.search("\\b(in)_", words[j-1], re.IGNORECASE) and re.search("\\b(most)_", words[j], re.IGNORECASE) and re.search("\\b(cases|instances)_", words[j+1], re.IGNORECASE)):
                words[j] = re.sub("_(\w+)", "_\\1 HDG", words[j])
            #---------------------------------------------------

            # Tags politeness markers
            # ELF new variables for: thanks, thank you, ta, please, mind_VB, excuse_V, sorry, apology and apologies.
            if ((re.search("\\bthank_", words[j], re.IGNORECASE) and re.search("\\byou", words[j+1], re.IGNORECASE)) or
            (re.search("\\bsorry_|\\bexcuse_V|\\bapology_|\\bapologies_|\\bplease_|\\bcheers_", words[j], re.IGNORECASE)) or
            (re.search("\\bthanks_", words[j], re.IGNORECASE) and not re.search("\\bto_", words[j+1], re.IGNORECASE)) or # Avoids the confusion with the conjunction "thanks to"
            (not re.search("\\bgot_", words[j-1], re.IGNORECASE) and re.search("\\bta_", words[j], re.IGNORECASE)) or # Avoids confusion with gotta
            (re.search("\\bI_|\\bwe_", words[j-2], re.IGNORECASE) and re.search("\\b(" + be + ")", words[j-1], re.IGNORECASE) and re.search("\\bwonder_V|\\bwondering_", words[j], re.IGNORECASE)) or
            (re.search("\\byou_|_XX0", words[j-1], re.IGNORECASE) and re.search("\\bmind_V", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_\\1 POLITE", words[j])

            # Tags HAVE GOT constructions
            # ELF: New variable. Added a tag for "have got" constructions, overriding the PEAS and PASS constructions.
            if (re.search("\\bgot", words[j], re.IGNORECASE)):

                if ((re.search("\\b(" + have + ")", words[j-1], re.IGNORECASE)) or # have got
                (re.search("_RB|_XX0|_EMPH|_DMA", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE)) or # have not got
                (re.search("_RB|_XX0|_EMPH|_DMA", words[j-1]) and re.search("_RB|_XX0|_EMPH|_DMA", words[j-2]) and re.search("\\b(" + have + ")", words[j-3], re.IGNORECASE)) or # haven't they got
                (re.search("_NN|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE)) or # has he got?
                (re.search("_XX0|_RB|_EMPH|_DMA", words[j-1]) and re.search("_NN|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-2]) and re.search("\\b(" + have + ")", words[j-3], re.IGNORECASE))): # hasn't he got?
                    words[j] = re.sub("_\w+", "_HGOT", words[j])
            
                if (re.search("\\b(" + have + ")", words[j-1], re.IGNORECASE) and re.search("_VBD|_VBN", words[j+1])):
                    words[j] = re.sub("_(\w+)", "_PEAS", words[j])
                    words[j+1] = re.sub("_(\w+)", "_PGET", words[j+1])
                # Correction for: she has got arrested

                if (re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE) and re.search("_RB|_XX0|_EMPH|_DMA", words[j-1], re.IGNORECASE) and re.search("_VBD|_VBN", words[j+1])):
                    words[j] = re.sub("_(\w+)", "_PEAS", words[j])
                    words[j+1] = re.sub("_(\w+)", "_PGET", words[j+1])
                    # Correction for: she hasn't got arrested

            #---------------------------------------------------

    # EVEN MORE COMPLEX TAGS

    for j, value in enumerate(words):
        
        if value != " ":
      
            #---------------------------------------------------    

            # Tags WH questions
            # ELF: rewrote this new operationalisation because Biber/Nini's code relied on a full stop appearing before the question word. 
            # This new operationalisation requires a question word (from a much shorter list taken from the COBUILD that Nini's/Biber's list) that is not followed by another question word and then a question mark within 15 words. 

            if ((re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_", words[j+1]))  or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j+1], re.IGNORECASE) and re.search("\\?_\\.", words[j+2])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j+1], re.IGNORECASE) and re.search("\\?_\\.", words[j+3])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\b(" + be + ")|\\b(" + have + ")|\\b(" + do + ")|_MD", words[j+1], re.IGNORECASE) and re.search("\\?_\\.", words[j+4])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+2])) or # Who cares?
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+3])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+4])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+5])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+6])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+7])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+8])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+9])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+10])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+11])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+12])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+13])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+14])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+15])) or
            (re.search("\\b(" + whw + ")", words[j], re.IGNORECASE) and re.search("\\?_\\.$", words[j+16]))):
                words[j] = re.sub("(\w+)_\w+", "\\1_WHQU", words[j])

            if ((re.search("\\b(" + whw + ")", words[j-1], re.IGNORECASE) and re.search("\\?_", words[j]))  or
            (re.search("_WHQU", words[j-2]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-3]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-4]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-5]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-6]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-7]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-8]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-9]) and re.search("\\?_\\.", words[j])) or
            (re.search("_WHQU", words[j-10]) and re.search("\\?_\\.", words[j])) or        
            (re.search("_WHQU", words[j-11]) and re.search("\\?_\\.", words[j])) or 
            (re.search("_WHQU", words[j-12]) and re.search("\\?_\\.", words[j])) or        
            (re.search("_WHQU", words[j-13]) and re.search("\\?_\\.", words[j])) or        
            (re.search("_WHQU", words[j-14]) and re.search("\\?_\\.", words[j])) or        
            (re.search("_WHQU", words[j-15]) and re.search("\\?_\\.", words[j]))):        
                words[j] = re.sub("_\\.$", "_. WQ", words[j]) # This line will add a dummy WQ tag (will not be counted in the final tables of counts) to allow for remaining question marks (except those immediately preceeded by an FPUH to be assigned the YNQU tag)
                words[j] = re.sub("_\\. YNQU", "_. WQ", words[j]) # This line should erase any YNQU tags that have (probably) been wrongly assigned
                words[j] = re.sub("_\\. QUTAG", "_. WQ", words[j]) # This line should erase any QUTAG tags that have (probably) been wrongly assigned
            
        #---------------------------------------------------

            # Tags remaining attribute adjectives (JJAT)
            if ((re.search("_JJAT", words[j-2]) and re.search("\\band_", words[j-1], re.IGNORECASE) and re.search("_JJ", words[j])) or
            (re.search("_JJ", words[j]) and re.search("\\band_", words[j+1], re.IGNORECASE) and re.search("_JJAT", words[j+2])) or
            (re.search("_JJAT", words[j-2]) and re.search(",_,", words[j-1]) and re.search("_JJ", words[j])) or
            (re.search("_JJ", words[j]) and re.search(",_,", words[j+1]) and re.search("_JJAT", words[j+2])) or
            (re.search("_JJ", words[j]) and re.search(",_,", words[j+1]) and re.search("\\band_", words[j+2]) and re.search("_JJAT", words[j+3])) or
            (re.search("_JJAT", words[j-3]) and re.search(",_,", words[j-2]) and re.search("\\band_", words[j-1]) and re.search("_JJ", words[j]))):
                words[j] = re.sub("_\w+", "_JJAT", words[j])

            #---------------------------------------------------

            # Tags perfect aspects 
            # ELF: Changed things around as compared to the MAT to tag PEAS onto the past participle (and thus replace the VBD/VBN tags) rather than as an add-on to the verb have, as Biber/Nini did. 
            # I tried to avoid as many errors as possible with 's being either BE (= passive) or HAS (= perfect aspect) but this is not perfect. 
            # Note that "'s got" and "'s used to" are already tagged separately. 
            # It's also worth noting that lemmatisation would likely not help much here either because spot checks with Sketch Engine's lemmatiser show that lemmatisers do a terrible job at this, too!
            if ((re.search("ed_VBD|_VBN", words[j]) and re.search("\\b(" + have + ")", words[j-1], re.IGNORECASE)) or # have eaten
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_RB|_XX0|_EMPH|_PRP|_DMA|_CC", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE)) or # have not eaten
            (re.search("\\bbeen_PASS|\\bhad_PASS|\\bdone_PASS|\\b(" + v_stative + ")_PASS", words[j], re.IGNORECASE) and re.search("\\bs_VBZ", words[j-1], re.IGNORECASE)) or # This ensures that 's + past participle combinations which are unlikely to be passives are overwritten here as PEAS
            (re.search("\\bbeen_PASS|\\bhad_PASS|\\bdone_PASS|\\b(" + v_stative + ")_PASS", words[j], re.IGNORECASE) and re.search("_RB|_XX0|_EMPH|_DMA", words[j-1]) and re.search("\\bs_VBZ", words[j-2], re.IGNORECASE)) or # This ensures that 's + not/ADV + past participle combinations which are unlikely to be passives are overwritten here as PEAS
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_RB|_XX0|_EMPH|_CC", words[j-2]) and re.search("\\b(" + have + ")", words[j-3], re.IGNORECASE)) or # haven't really eaten, haven't you noticed?
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_NN|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE)) or # has he eaten?
            (re.search("\\b(" + have + ")", words[j-1], re.IGNORECASE) and re.search("ed_VBD|_VBN", words[j]) and re.search("_P", words[j+1])) or # has been told or has got arrested
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_P", words[j+1]) and re.search("_XX0|_RB|_EMPH|_DMA|_CC", words[j-1]) and re.search("_XX0|_RB|_EMPH", words[j-2]) and re.search("\\b(" + have + ")", words[j-3], re.IGNORECASE)) or #hasn't really been told
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_PASS", words[j+1]) and re.search("_XX0|_RB|_EMPH|_DMA|_CC", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE)) or # hasn't been told
            (re.search("ed_VBD|_VBN", words[j]) and re.search("_XX0|_EMPH|_DMA|_CC", words[j+1]) and re.search("_NN|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("\\b(" + have + ")", words[j-2], re.IGNORECASE))): # hasn't he eaten?
                    words[j] = re.sub("_\w+", "_PEAS", words[j])

            # This corrects some of the 'd wrongly identified as a modal "would" by the Stanford Tagger 
            if (re.search("'d_MD", words[j-1], re.IGNORECASE) and re.search("_VBN", words[j])): # He'd eaten
                words[j-1] = re.sub("_\w+", "_VBD", words[j-1])
                words[j] = re.sub("_\w+", "_PEAS", words[j])

            if (re.search("'d_MD", words[j-1], re.IGNORECASE) and re.search("_RB|_EMPH", words[j]) and re.search("_VBN", words[j+1])): # She'd never been
                words[j-1] = re.sub("_\w+", "_VBD", words[j-1])
                words[j+1] = re.sub("_\w+", "_PEAS", words[j+1])

            # This corrects some of the 'd wrongly identified as a modal "would" by the Stanford Tagger 
            if (re.search("\\bbetter_", words[j]) and re.search("'d_MD", words[j-1], re.IGNORECASE)):
                words[j-1] = re.sub("_\w+", "_VBD", words[j-1])

            if (re.search("_VBN|ed_VBD|en_VBD", words[j]) and re.search("\\band_|\\bor_", words[j-1], re.IGNORECASE) and re.search("_PASS", words[j-2])): # This accounts for the second passive form in phrases such as "they were selected and extracted"
                words[j-1] = re.sub("_\w+", "_CC", words[j-1]) # OR _PHC if this variable is used! (see problems described in tagger performance evaluation)
                words[j] = re.sub("_\w+", "_PASS", words[j])

            # ELF: Added a "used to" variable, overriding the PEAS and PASS constructions. 
            # Not currently in use due to very low precision (see tagger performance evaluation).
            #if (re.search("\\bused_", words[j], re.IGNORECASE) and re.search("\\bto_", words[j+1])):
            # words[j] = re.sub("_\w+", "_USEDTO", words[j])
            #

            # ELF: tags "able to" constructions. New variable
            if ((re.search("\\b(" + be + ")", words[j-1]) and re.search("\\bable_J|\\bunable_J", words[j], re.IGNORECASE) and re.search("\\bto_", words[j+1])) or
            (re.search("\\b(" + be + ")", words[j-2]) and re.search("\\bable_J|\\bunable_J", words[j], re.IGNORECASE) and re.search("\\bto_", words[j+1]))):
                words[j-1] = re.sub("_(\w+)", "_\\1 BEMA", words[j-1])
                words[j] = re.sub("_\w+", "_ABLE", words[j])

        #---------------------------------------------------

    # ELF: added tag for the progressive aspect
    # Note that it's important that this tag has its own loop because it relies on GTO (going to + inf. constructions) having previously been tagged. 
    # Note that this script overrides the _VBG Stanford tagger tag so that the VBG count now includes all (non-finite) -ing constructions *except* progressives and GOING-to constructions.

    for j, value in enumerate(words):

        if value != " ":

            if (re.search("_VBG", words[j])):
                if ((re.search("\\b(" + be + ")", words[j-1], re.IGNORECASE)) or # am eating
                (re.search("_RB|_XX0|_EMPH|_CC", words[j-1]) and re.search("\\b(" + be + ")|'m_V", words[j-2], re.IGNORECASE)) or # am not eating
                (re.search("_RB|_XX0|_EMPH|_CC", words[j-1]) and re.search("_RB|_XX0|_EMPH|_CC", words[j-2]) and re.search("\\b(" + be + ")", words[j-3], re.IGNORECASE)) or # am not really eating
                (re.search("_NN|_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("\\b(" + be + ")", words[j-2], re.IGNORECASE)) or # am I eating
                (re.search("_NN|_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-1]) and re.search("_XX0|_EMPH", words[j-2]) and re.search("\\b(" + be + ")", words[j-3], re.IGNORECASE)) or # aren't I eating?
                (re.search("_XX0|_EMPH", words[j-1]) and re.search("_NN|_PRP|\\bi_|\\bwe_|\\bhe_|\\bshe_|\\bit_P|\\bthey_", words[j-2]) and re.search("\\b(" + be + ")", words[j-3], re.IGNORECASE))): # am I not eating
                    words[j] = re.sub("_\w+", "_PROG", words[j])

            #---------------------------------------------------

            # ELF: Added a new variable for quotative uses of like (BE + like)
            # ELF: However, QLIKE is currently not in use due to relatively low precision and recall (see MFTE perl performance evaluation).
            
            #if (re.search("\\b(" + be + ")", words[j-1]) and re.search("\\blike_IN|\\blike_JJ", words[j], re.IGNORECASE) and not re.search("_NN|_J|_DT|_\.|_,|_IN", words[j+1])):
            #words[j] = re.sub("_\w+", "_QLIKE", words[j])

            #---------------------------------------------------

            # ELF: Added a "bin" variable for "like" 
            # Because attempts to disambiguate between like as a preposition (IN) and adjective (JJ) largely failed. 
            # In conversation, like also frequently acts as a filler or interjection or as part of the quotative phrase BE+like. 
            # Recall and precision were too low (see MFTE perl evaluation) 
            
            if (re.search("\\blike_IN|\\blike_JJ|\\blike_UH", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_LIKE", words[j])

    #---------------------------------------------------

    # Tags be as main verb ELF: Ensured that question tags are not being assigned this tag by adding the exceptions of QUTAG occurrences.

    for j, value in enumerate(words):

        if value != " ":

            if ((not re.search("_EX", words[j-2]) and not re.search("_EX", words[j-1]) and re.search("\\b(" + be + ")|\\bbeen_", words[j], re.IGNORECASE) and re.search("_CD|_DT|_PRP|_J|_IN|_QUAN|_EMPH|_CUZ|\\b(" + whw + ")", words[j+1]) and not re.search("QUTAG|_PROG", words[j+2]) and not re.search("QUTAG|_PROG", words[j+3])) or

            (not re.search("_EX", words[j-2]) and not re.search("_EX", words[j-1]) and re.search("\\b(" + be + ")|\\bbeen_", words[j], re.IGNORECASE) and not re.search("_V", words[j+1]) and re.search("\W+_", words[j+2]) and not re.search(" QUTAG", words[j+2])) or # Who is Dinah? Ferrets are ferrets!

            (not re.search("_EX", words[j-2]) and not re.search("_EX", words[j-1]) and re.search("\\b(" + be + ")", words[j], re.IGNORECASE) and re.search("_XX0|_NN", words[j+2]) and not re.search("_V", words[j+2]) and re.search("\W+_", words[j+3]) and not re.search(" QUTAG", words[j+3])) or # London is not Paris.

            (not re.search("_EX", words[j-2]) and not re.search("_EX", words[j-1]) and re.search("\\b(" + be + ")|\\bbeen_", words[j], re.IGNORECASE) and re.search("_CD|_DT|_PRP|_J|_IN|_QUAN|_RB|_EMPH|_NN", words[j+1]) and re.search("_CD|_DT|_PRP|_J|_IN|_QUAN|to_TO|_EMPH", words[j+2]) and not re.search("QUTAG|_PROG|_PASS", words[j+2]) and not re.search("QUTAG|_PROG|_PASS", words[j+3]) and not re.search(" QUTAG|_PROG|_PASS", words[j+4])) or # She was so much frightened

            (not re.search("_EX", words[j-2]) and not re.search("_EX", words[j-1]) and re.search("\\b(" + be + ")|\\bbeen_", words[j], re.IGNORECASE) and re.search("_RB|_XX0", words[j+1]) and re.search("_CD|_DT|_PRP|_J|_IN|_QUAN|_EMPH", words[j+2]) and not re.search(" QUTAG|_PROG|_PASS", words[j+2]) and not re.search(" QUTAG|_PROG|_PASS", words[j+3]))):

                words[j] = re.sub("_(\w+)", "_\\1 BEMA", words[j])

    #---------------------------------------------------
    # Tags demonstratives 
    # ELF: New, much simpler variable. Also corrects any leftover "that_IN" and "that_WDT" to DEMO. 
    # These have usually been falsely tagged by the Stanford Tagger, especially they end sentences, e.g.: Who did that?

    for j, value in enumerate(words):

        if value != " ":

            if (re.search("\\bthat_DT|\\bthis_DT|\\bthese_DT|\\bthose_DT|\\bthat_IN|\\bthat_WDT", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_DEMO", words[j])
            
            if (re.search(" BEMA", words[j-1]) and re.search("_J", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_JJPR", words[j])

    #---------------------------------------------------
    # Tags subordinator "that" omission 
    # ELF: Added words+2 in the first pattern to remove "Why would I know that?". 
    # Replaced the long MD/do/have/be/V regex that had a lot of redundancies by just MD/V. 
    # In the second pattern, replaced all PRPS by just subject position ones to remove phrases like "He didn't hear me thank God". 
    # Originally also added the pronoun "it" which Nini had presumably forgotten? Then simply used the PRP tag for all personal pronouns.

    for j, value in enumerate(words):

        if value != " ":

            if ((re.search("\\b((" + public + ")|(" + private + ")|(" + suasive + "))", words[j], re.IGNORECASE) and re.search("_DEMO|_PRP|_N", words[j+1]) and re.search("_MD|_V|_DT", words[j+2])) or

            (re.search("\\b((" + public + ")|(" + private + ")|(" + suasive + "))", words[j], re.IGNORECASE) and re.search("_J|_RB|_DT|_QUAN|_CD|_PRP", words[j+1]) and re.search("_N", words[j+2]) and re.search("_MD|_V", words[j+3])) or

            (re.search("\\b((" + public + ")|(" + private + ")|(" + suasive + "))", words[j], re.IGNORECASE) and re.search("_J|_RB|_DT|_QUAN|_CD|_PRP", words[j+1]) and re.search("_J", words[j+2]) and re.search("_N", words[j+3]) and re.search("_MD|_V", words[j+4]))):

                words[j] = re.sub("_(\w+)", "_\\1 THATD", words[j])

    #---------------------------------------------------

    # Tags pronoun "it" 
    # ELF: excluded IT (all caps) from the list since it usually refers to information technology

    for j, value in enumerate(words):  

        if value != " ":
            if ((re.search("\\bits_|\\bitself_", words[j], re.IGNORECASE)) or
            (re.search("\\bit_|\\bIt_|\\bIT_P", words[j]))):
                words[j] = re.sub("_\w+", "_PIT", words[j])

        #---------------------------------------------------

    # Tags first person references 
    # ELF: Added exclusion of occurrences of US (all caps) which usually refer to the United States.
    # ELF: Added 's_PRP to account for abbreviated "us" in "let's" 
    # Also added: mine and ours.

    # ELF: Later subdivided Biber's FPP1 into singular and plural.

    for j, value in enumerate(words):

        if value != " ":

            if (re.search("\\bI_P|\\bme_|\\bmy_|\\bmyself_|\\bmine_|\\bi_SYM|\\bi_FW", words[j], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_PP1S", words[j])


            if ((re.search("\\bwe_|\\bour_|\\bourselves_|\\bours_|\\b's_PRP", words[j], re.IGNORECASE)) or
            (re.search("\\bus_P|\\bUs_P", words[j]))):
                words[j] = re.sub("_\w+", "_PP1P", words[j])

            if (re.search("\\blet_", words[j], re.IGNORECASE) and re.search("'s_|\\bus_", words[j+1], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_VIMP", words[j])
                words[j+1] = re.sub("_\w+", "_PP1P", words[j+1])

            if (re.search("\\blet_", words[j], re.IGNORECASE) and re.search("\\bme_", words[j+1], re.IGNORECASE)):
                words[j] = re.sub("_\w+", "_VIMP", words[j])
                words[j+1] = re.sub("_\w+", "_PP1S", words[j+1])

    #---------------------------------------------------

    for j, value in enumerate(words):

        if value != " ":

        # Tags concessive conjunctions 
        # Note that Nini had already added "THO" to Biber's list.
        # ELF added: despite, albeit, yet, except that, in spite of, granted that, granted + punctuation, no matter + WH-words, regardless of + WH-word. 
        # Also added: nevertheless, nonetheless and notwithstanding and whereas, which Biber had as "other adverbial subordinators" (OSUB, a category ELF removed).
            if ((re.search("\\balthough_|\\btho_|\\bdespite|\\balbeit_|nevertheless_|nonetheless_|notwithstanding_|\\bwhereas_", words[j], re.IGNORECASE)) or
            (re.search("\\bexcept_", words[j], re.IGNORECASE) and re.search("\\bthat_", words[j+1], re.IGNORECASE)) or    	
            (re.search("\\bgranted_", words[j], re.IGNORECASE) and re.search("\\bthat_|_,", words[j+1], re.IGNORECASE)) or		
            (re.search("\\bregardless_|\\birregardless_", words[j], re.IGNORECASE) and re.search("\\bof_", words[j+1], re.IGNORECASE)) or
            (re.search("\\beven_", words[j-1], re.IGNORECASE) and re.search("\\bif_", words[j], re.IGNORECASE)) or
            (re.search("\\byet_|\\bstill_", words[j], re.IGNORECASE) and re.search("_,", words[j+1], re.IGNORECASE)) or
            (not re.search("\\bas_", words[j-1], re.IGNORECASE) and re.search("\\bthough_", words[j], re.IGNORECASE)) or
            (re.search("\\byet_|\\bgranted_|\\bstill_", words[j], re.IGNORECASE) and re.search("_\W", words[j-1], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_CONC", words[j])

            if ((re.search("\\bno_", words[j-1], re.IGNORECASE) and re.search("\\bmatter_", words[j], re.IGNORECASE) and re.search("\\b(" + whw + ")", words[j+1], re.IGNORECASE)) or
            (re.search("\\bin_", words[j-1], re.IGNORECASE) and re.search("\\bspite_", words[j]) and re.search("\\bof_", words[j+1]))):
                words[j] = re.sub("_(\w+)", "_\\1 CONC", words[j])

            #---------------------------------------------------

            # Tags place adverbials 
            # ELF: added all the words from "downwind" onwards and excluded "there" tagged as an existential "there" as in "there are probably lots of bugs in this script". Also restricted above, around, away, behind, below, beside, inside and outside to adverb forms only.
            if (re.search("\\baboard_|\\babove_RB|\\babroad_|\\bacross_RB|\\bahead_|\\banywhere_|\\balongside_|\\baround_RB|\\bashore_|\\bastern_|\\baway_RB|\\bbackwards?|\\bbehind_RB|\\bbelow_RB|\\bbeneath_|\\bbeside_RB|\\bdownhill_|\\bdownstairs_|\\bdownstream_|\\bdownwards_|\\beast_|\\bhereabouts_|\\bindoors_|\\binland_|\\binshore_|\\binside_RB|\\blocally_|\\bnear_|\\bnearby_|\\bnorth_|\\bnowhere_|\\boutdoors_|\\boutside_RB|\\boverboard_|\\boverland_|\\boverseas_|\\bsouth_|\\bunderfoot_|\\bunderground_|\\bunderneath_|\\buphill_|\\bupstairs_|\\bupstream_|\\bupwards?|\\bwest_|\\bdownwind|\\beastwards?|\\bwestwards?|\\bnorthwards?|\\bsouthwards?|\\belsewhere|\\beverywhere|\\bhere_|\\boffshore|\\bsomewhere|\\bthereabouts?|\\bfar_RB|\\bthere_RB|\\bonline_|\\boffline_N", words[j], re.IGNORECASE) 
            and not re.search("_NNP", words[j])):
                words[j] = re.sub("_\w+", "_PLACE", words[j])

            if (re.search("\\bthere_P", words[j], re.IGNORECASE) and re.search("_MD", words[j+1])): # Correction of there + modals, e.g. there might be that option which are frequently not recognised as instances of there_EX by the Stanford Tagger
                words[j] = re.sub("_\w+", "_EX", words[j])

            #---------------------------------------------------
            # Tags time adverbials 
            # ELF: Added already, so far, thus far, yet (if not already tagged as CONC above) and ago. Restricted after and before to adverb forms only.
            if ((re.search("\\bago_|\\bafter_RB|\\bafterwards_|\\bagain_|\\balready_|\\bbefore_RB|\\bbeforehand_|\\bbriefly_|\\bcurrently_|\\bearlier_RB|\\bearly_RB|\\beventually_|\\bformerly_|\\bimmediately_|\\binitially_|\\binstantly_|\\bforeever_|\\blate_RB|\\blately_|\\blater_RB|\\bmomentarily_|\\bnow_|\\bnowadays_|\\bonce_|\\boriginally_|\\bpresently_|\\bpreviously_|\\brecently_|\\bshortly_|\\bsimultaneously_|\\bsooner_|\\bsubsequently_|\\bsuddenly|\\btoday_|\\bto-day_|\\btomorrow_|\\bto-morrow_|\\btonight_|\\bto-night_|\\byesterday_|\\byet_RB|\\bam_RB|\\bam_NN|\\bpm_NN|\\bpm_RB", words[j], re.IGNORECASE)) or
            (re.search("\\bsoon_", words[j], re.IGNORECASE) and not re.search("\\bas_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bprior_", words[j], re.IGNORECASE) and re.search("\\bto_", words[j+1], re.IGNORECASE)) or
            (re.search("\\bso_|\\bthus_", words[j-1], re.IGNORECASE) and re.search("\\bfar_", words[j], re.IGNORECASE) and not re.search("_J|_RB", words[j+1], re.IGNORECASE))):
                words[j] = re.sub("_\w+", "_TIME", words[j])

            #---------------------------------------------------

    for j, value in enumerate(words):
        if (re.search("\\b(" + do + ")", words[j], re.IGNORECASE) and not re.search(" DOAUX", words[j])):
            words[j] = re.sub("_(\w+)", "_\\1 ACT", words[j])
            #print(words[j])

        try:
            # Adds "NEED to" and "HAVE to" to the list of necessity (semi-)modals  
            if (re.search("\\bneed_V|\\bneeds_V|\\bneeded_V|\\bhave_V|\\bhas_V|\\bhad_V|\\bhaving_V", words[j], re.IGNORECASE) and re.search("\\bto_TO", words[j+1])):
                words[j] = re.sub("_(\w+)", "_MDNE", words[j])
        except IndexError:
            continue

        # EFL: Tag remaining YNQU questions
        # This loop assumes that any question marks that have yet to be tagged as YNQU, WHQU or QUTAG by this stage are in fact yes-no questions.
        if (re.search("\\?_\\.", words[j]) and not re.search("YNQU|WQ|QUTAG", words[j]) and not re.search("_UH|_FPUH|_DMA", words[j-1])):
            words[j] = re.sub("(_\\.)", "\\1 YNQU", words[j]) 
        
    #--------------------------------------------------- 
    # BASIC TAGS THAT HAVE TO BE TAGGED AT THE END TO AVOID CLASHES WITH MORE COMPLEX REGEX ABOVE
    for index, x in enumerate(words):

        if x != " ":

        # Tags amplifiers 
            if (re.search("\\babsolutely_|\\baltogether_|\\bdefinitely_|\\bcompletely_|\\benormously_|\\bentirely_|\\bespecially_|\\bextremely_|\\bextraordinarily_|\\bfully_|\\bgreatly_|\\bhighly_|\\bintensely_|\\bperfectly_|\\bsorely_|\\bstrongly_|\\bthoroughly_|\\btotally_|\\butterly_|\\bvery_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_AMP", words[index])

            # Tags downtoners
            # ELF: Added "less" as an adverb (note that "less" as an adjective is tagged as a quantifier further up)
            if (re.search("\\balmost_|\\bbarely_|\\bhardly_|\\bless_JJ|\\bmerely_|\\bmildly_|\\bnearly_|\\bonly_|\\bpartially_|\\bpartly_|\\bpractically_|\\bscarcely_|\\bslightly_|\\bsomewhat_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_DWNT", words[index])

            # Corrects EMO tags
            # ELF: Correction of emoticon issues to do with the Stanford tags for brackets including hyphens
            if (re.search("_EMO(.)*-", words[index], re.IGNORECASE)):
                words[index] = re.sub("_EMO(.)*-", "_EMO", words[index])

            # Tags quantifier references 
            # ELF: Added any, removed nowhere (which is now place). "no one" is also tagged for at an earlier stage to avoid collisions with the XX0 variable.
            if (re.search("\\banybody_|\\banyone_|\\banything_|\\beverybody_|\\beveryone_|\\beverything_|\\bnobody_|\\bnone_|\\bnothing_|\\bsomebody_|\\bsomeone_|\\bsomething_|\\bsomewhere|\\bnoone_|\\bno-one_|\\bothers_", words[index], re.IGNORECASE)):      
                words[index] = re.sub("_\w+", "_QUPR", words[index])

            # Tags gerunds 
            # ELF: Not currently in use because of doubts about the usefulness of this category (cf. Herbst 2016 in Applied Construction Grammar) + high rate of false positives with Biber's/Nini's operationalisation of the variable.
            #if ((re.search("ing_NN", words[index], re.IGNORECASE) and re.search("\w{10,}", words[index])) or
            # (re.search("ings_NN", words[index], re.IGNORECASE) and re.search("\w{11,}", words[index]))):
                #words[index] = re.sub("_\w+", "_GER", words[index])

            # ELF added: pools together all proper nouns (singular and plural). Not currently in use since no distinction is made between common and proper nouns.
            #if (re.search("_NNPS", words[index])):
                # words[index] = re.sub("_\w+", "_NNP", words[index])

            # Tags predicative adjectives (JJPR) by joining all kinds of JJ (but not JJAT, see earlier loop)
            if (re.search("_JJS|_JJR|_JJ\\b", words[index])):
                words[index] = re.sub("_\w+", "_JJPR", words[index])

            # Tags total adverbs by joining all kinds of RB (but not those already tagged as HDG, FREQ, AMP, DWNTN, EMPH, ELAB, EXTD, TIME, PLACE...).
            if (re.search("_RBS|_RBR|_WRB", words[index])):
                words[index] = re.sub("_\w+", "_RB", words[index])

            # Tags present tenses
            if (re.search("_VBP|_VBZ", words[index])):
                words[index] = re.sub("_\w+", "_VPRT", words[index])

            # Tags second person references - ADDED "THOU", "THY", "THEE", "THYSELF" ELF: added nominal possessive pronoun (yours), added u, ur, ye and y' (for y'all).
            if (re.search("\\byou_|\\byour_|\\byourself_|\\byourselves_|\\bthy_|\\bthee_|\\bthyself_|\\bthou_|\\byours_|\\bur_|\\bu_PRP|\\bye_PRP|\\by'_|\\bthine_|\\bya_PRP", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_PP2", words[index])

            # Tags third person reference 
            # ELF: added themself in singular (cf. https://www.lexico.com/grammar/themselves-or-themself), added nominal possessive pronoun forms (hers, theirs), also added em_PRP for 'em.
            # ELF: Subdivided Biber's category into "they" references (PP3t), "she" references (PP3f) and "he" references (PP3m).
            if (re.search("\\bthey_|\\bthem_|\\btheir_|\\bthemselves_|\\btheirs_|\W+em_PRP|\\bthemself_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_PP3t", words[index])

            if (re.search("\\bshe_|\\bher_|\\bhers_|\\bherself_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_PP3f", words[index])

            if (re.search("\\bhe_|\\bhim_|\\bhis_|\\bhimself_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_PP3m", words[index])

            # Tags "can" modals 
            # ELF: added _MD onto all of these. And ca_MD which was missing for can't.
            if (re.search("\\bcan_MD|\\bca_MD|\\bcannot_", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDCA", words[index])

            # Tags "could" modals
            if (re.search("\\bcould_MD", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDCO", words[index])

            # Tags necessity modals
            # ELF: added _MD onto all of these to increase precision.
            if (re.search("\\bought_MD|\\bshould_MD|\\bmust_MD|\\bneed_MD", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDNE", words[index])

            # Tags "may/might" modals
            # ELF: added _MD onto all of these to increase precision.
            if (re.search("\\bmay_MD|\\bmight_MD", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDMM", words[index])

            # Tags will/shall modals. 
            # ELF: New variable replacing Biber's PRMD.
            if (re.search("\\bwill_MD|'ll_MD|\\bshall_|\\bsha_|\W+ll_MD", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDWS", words[index])

            # Tags would as a modal. 
            # ELF: New variable replacing PRMD.
            if (re.search("\\bwould_|'d_MD|\\bwo_MD|\\bd_MD|\W+d_MD", words[index], re.IGNORECASE)):
                words[index] = re.sub("_\w+", "_MDWO", words[index])

            #----------------------------------------
            # Tags verbal contractions
            if (re.search("'\w+_V|\\bn't_XX0|'ll_|'d_|\\bnt_XX0", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 CONT", words[index])

            # tags the remaining interjections and filled pauses. 
            # ELF: added variable
            # Note: it is important to keep this variable towards the end because some UH tags need to first be overridden by other variables such as politeness (please) and pragmatic markers (yes). 
            if (re.search("_UH", words[index])):
                words[index] = re.sub("_(\w+)", "_FPUH", words[index])

            # ELF: added variable: tags adverbs of frequency (list from COBUILD p. 270 but removed "mainly").
            if (re.search("\\busually_|\\balways_|\\boften_|\\bgenerally|\\bnormally|\\btraditionally|\\bagain_|\\bconstantly|\\bcontinually|\\bfrequently|\\bever_|\\bnever_|\\binfrequently|\\bintermittently|\\boccasionally|\\boftens_|\\bperiodically|\\brarely_|\\bregularly|\\brepeatedly|\\bseldom|\\bsometimes|\\bsporadically|daily_|weekly_|yearly", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_FREQ", words[index])

            # ELF: remove the TO category which was needed for the identification of other features put overlaps with VB
            if (re.search("_TO", words[index])):
                words[index] = re.sub("_(\w+)", "_IN", words[index])

        #---------------------------------------------------

        # Tags noun compounds 
        # ELF: New variable. Only works to reasonable degree of accuracy with "well-punctuated" (written) language, though.
        # Allows for the first noun to be a proper noun but not the second thus allowing for "Monday afternoon" and "Hollywood stars" but not "Barack Obama" and "L.A.". Also restricts to nouns with a minimum of two letters to avoid OCR errors (dots and images identified as individual letters and which are usually tagged as nouns) producing lots of NCOMP's.
    
    for j, value in enumerate(words):

        if value != " ":

            # Shakir: Added space to prevent tag overlaps
            if (re.search("\\b.{2,}_NN", words[j]) and re.search("\\b(.{2,}_NN|.{2,}_NNS)\\b", words[j+1]) and not re.search("NCOMP", words[j]) and not re.search(" ", words[j])):
                words[j+1] = re.sub("_(\w+)", "_\\1 NCOMP", words[j+1])

            # Shakir: if extended is True keep proper noun distinction
            if extended:

                if (re.search("_NNP|_NNPS", words[j])):
                    
                    words[j] = re.sub("_\w+", "_NN NNP", words[j])

            # Tags total nouns by joining plurals together with singulars including of proper nouns.            
            if (re.search("_NN|_NNS|_NNP|_NNPS", words[j])):
                
                words[j] = re.sub("_\w+", "_NN", words[j])

            # Shakir: fixed it tagged as PRP 
            if (re.search("(It|its?|itself)_PRP\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_PIT", words[j])

            # Shakir: additional variable to catch any remaining MD tags
            if (re.search("_MD\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_MDother", words[j])
            
            # Shakir: additional variable to catch any remaining PRPS and PRP tags
            if (re.search("_(PRPS|PRP)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_PPother", words[j])        

    return words

def process_sentence_extended (words: list) -> list:
    """Returns words list tagged with Biber's (2006) additional semantic categories
    Args:
        words (list): list of tagged words
    Returns:
        words (list): list of tagged words with tags applied
    """
    # DICTIONARY LISTS

    # The following lists are based on the verb semantic categories used in Biber 2006.
    
    # Activity verbs 
    # ELF: removed GET and GO due to high polysemy and corrected the "evercise" typo found in Biber 2006.
    vb_act = "(buy|buys|buying|bought|make|makes|making|made|give|gives|giving|gave|given|take|takes|taking|took|taken|come|comes|coming|came|use|uses|using|used|leave|leaves|leaving|left|show|shows|showing|showed|shown|try|tries|trying|tried|work|works|wrought|worked|working|move|moves|moving|moved|follow|follows|following|followed|put|puts|putting|pay|pays|paying|paid|bring|brings|bringing|brought|meet|meets|met|play|plays|playing|played|run|runs|running|ran|hold|holds|holding|held|turn|turns|turning|turned|send|sends|sending|sent|sit|sits|sitting|sat|wait|waits|waiting|waited|walk|walks|walking|walked|carry|carries|carrying|carried|lose|loses|losing|lost|eat|eats|ate|eaten|eating|watch|watches|watching|watched|reach|reaches|reaching|reached|add|adds|adding|added|produce|produces|producing|produced|provide|provides|providing|provided|pick|picks|picking|picked|wear|wears|wearing|wore|worn|open|opens|opening|opened|win|wins|winning|won|catch|catches|catching|caught|pass|passes|passing|passed|shake|shakes|shaking|shook|shaken|smile|smiles|smiling|smiled|stare|stares|staring|stared|sell|sells|selling|sold|spend|spends|spending|spent|apply|applies|applying|applied|form|forms|forming|formed|obtain|obtains|obtaining|obtained|arrange|arranges|arranging|arranged|beat|beats|beating|beaten|check|checks|checking|checked|cover|covers|covering|covered|divide|divides|dividing|divided|earn|earns|earning|earned|extend|extends|extending|extended|fix|fixes|fixing|fixed|hang|hangs|hanging|hanged|hung|join|joins|joining|joined|lie|lies|lying|lay|lain|lied|obtain|obtains|obtaining|obtained|pull|pulls|pulling|pulled|repeat|repeats|repeating|repeated|receive|receives|receiving|received|save|saves|saving|saved|share|shares|sharing|shared|smile|smiles|smiling|smiled|throw|throws|throwing|threw|thrown|visit|visits|visiting|visited|accompany|accompanies|accompanying|accompanied|acquire|acquires|acquiring|acquired|advance|advances|advancing|advanced|behave|behaves|behaving|behaved|borrow|borrows|borrowing|borrowed|burn|burns|burning|burned|burnt|clean|cleaner|cleanest|cleans|cleaning|cleaned|climb|climbs|climbing|climbed|combine|combines|combining|combined|control|controls|controlling|controlled|defend|defends|defending|defended|deliver|delivers|delivering|delivered|dig|digs|digging|dug|encounter|encounters|encountering|encountered|engage|engages|engaging|engaged|exercise|exercised|exercising|exercises|expand|expands|expanding|expanded|explore|explores|exploring|explored|reduce|reduces|reducing|reduced)"

    # Communication verbs 
    # ELF: corrected a typo for "descibe" and added its other forms, removed "spake" as a form of SPEAK, removed some adjective forms like "fitter, fittest", etc.
    # In addition, British spellings and the verbs "AGREE, ASSERT, BEG, CONFIDE, COMMAND, DISAGREE, OBJECT, PLEDGE, PRONOUNCE, PLEAD, REPORT, TESTIFY, VOW" (taken from the public and suasive lists above) were added. "MEAN" which was originally assigned to the mental verb list was added to the communication list, instead.
    vb_comm = "(say|says|saying|said|tell|tells|telling|told|call|calls|calling|called|ask|asks|asking|asked|write|writes|writing|wrote|written|talk|talks|talking|talked|speak|speaks|spoke|spoken|speaking|thank|thanks|thanking|thanked|describe|describing|describes|described|claim|claims|claiming|claimed|offer|offers|offering|offered|admit|admits|admitting|admitted|announce|announces|announcing|announced|answer|answers|answering|answered|argue|argues|arguing|argued|deny|denies|denying|denied|discuss|discusses|discussing|discussed|encourage|encourages|encouraging|encouraged|explain|explains|explaining|explained|express|expresses|expressing|expressed|insist|insists|insisting|insisted|mention|mentions|mentioning|mentioned|offer|offers|offering|offered|propose|proposes|proposing|proposed|quote|quotes|quoting|quoted|reply|replies|replying|replied|shout|shouts|shouting|shouted|sign|signs|signing|signed|sing|sings|singing|sang|sung|state|states|stating|stated|teach|teaches|teaching|taught|warn|warns|warning|warned|accuse|accuses|accusing|accused|acknowledge|acknowledges|acknowledging|acknowledged|address|addresses|addressing|addressed|advise|advises|advising|advised|appeal|appeals|appealing|appealed|assure|assures|assuring|assured|challenge|challenges|challenging|challenged|complain|complains|complaining|complained|consult|consults|consulting|consulted|convince|convinces|convincing|convinced|declare|declares|declaring|declared|demand|demands|demanding|demanded|emphasize|emphasizes|emphasizing|emphasized|emphasise|emphasises|emphasising|emphasised|excuse|excuses|excusing|excused|inform|informs|informing|informed|invite|invites|inviting|invited|persuade|persuades|persuading|persuaded|phone|phones|phoning|phoned|pray|prays|praying|prayed|promise|promises|promising|promised|question|questions|questioning|questioned|recommend|recommends|recommending|recommended|remark|remarks|remarking|remarked|respond|responds|responding|responded|specify|specifies|specifying|specified|swear|swears|swearing|swore|sworn|threaten|threatens|threatening|threatened|urge|urges|urging|urged|welcome|welcomes|welcoming|welcomed|whisper|whispers|whispering|whispered|suggest|suggests|suggesting|suggested|plead|pleads|pleaded|pleading|agree|agrees|agreed|agreeing|assert|asserts|asserting|asserted|beg|begs|begging|begged|confide|confides|confiding|confided|command|commands|commanding|commanded|disagree|disagreeing|disagrees|disagreed|object|objects|objected|objects|pledge|pledges|pledging|pledged|report|reports|reported|reporting|testify|testifies|testified|testifying|vow|vows|vowing|vowed|mean|means|meaning|meant)"

    # Mental verbs
    # ELF: Added British spellings, removed AFFORD and FIND. Removed DESERVE which is also on Biber's (2006) existential list. Added wan to account for wanna tokenised as wan na.
    vb_mental =	"(see|sees|seeing|saw|seen|know|knows|knowing|knew|known|think|thinks|thinking|thought|want|wan|wants|wanting|wanted|need|needs|needing|needed|feel|feels|feeling|felt|like|likes|liking|liked|hear|hears|hearing|heard|remember|remembers|remembering|remembered|believe|believes|believing|believed|read|reads|reading|consider|considers|considering|considered|suppose|supposes|supposing|supposed|listen|listens|listening|listened|love|loves|loving|loved|wonder|wonders|wondering|wondered|understand|understands|understood|expect|expects|expecting|expected|hope|hopes|hoping|hoped|assume|assumes|assuming|assumed|determine|determines|determining|determined|agree|agrees|agreeing|agreed|bear|bears|bearing|bore|borne|care|cares|caring|cared|choose|chooses|choosing|chose|chosen|compare|compares|comparing|compared|decide|decides|deciding|decided|discover|discovers|discovering|discovered|doubt|doubts|doubting|doubted|enjoy|enjoys|enjoying|enjoyed|examine|examines|examining|examined|face|faces|facing|faced|forget|forgets|forgetting|forgot|forgotten|hate|hates|hating|hated|identify|identifies|identifying|identified|imagine|imagines|imagining|imagined|intend|intends|intending|intended|learn|learns|learning|learned|learnt|miss|misses|missing|missed|mind|minds|minding|notice|notices|noticing|noticed|plan|plans|planning|planned|prefer|prefers|preferring|preferred|prove|proves|proving|proved|proven|realize|realizes|realizing|realized|recall|recalls|recalling|recalled|recognize|recognizes|recognizing|recognized|recognise|recognises|recognising|recognised|regard|regards|regarding|regarded|suffer|suffers|suffering|suffered|wish|wishes|wishing|wished|worry|worries|worrying|worried|accept|accepts|accepting|accepted|appreciate|appreciates|appreciating|appreciated|approve|approves|approving|approved|assess|assesses|assessing|assessed|blame|blames|blaming|blamed|bother|bothers|bothering|bothered|calculate|calculates|calculating|calculated|conclude|concludes|concluding|concluded|celebrate|celebrates|celebrating|celebrated|confirm|confirms|confirming|confirmed|count|counts|counting|counted|dare|dares|daring|dared|detect|detects|detecting|detected|dismiss|dismisses|dismissing|dismissed|distinguish|distinguishes|distinguishing|distinguished|experience|experiences|experiencing|experienced|fear|fears|fearing|feared|forgive|forgives|forgiving|forgave|forgiven|guess|guesses|guessing|guessed|ignore|ignores|ignoring|ignored|impress|impresses|impressing|impressed|interpret|interprets|interpreting|interpreted|judge|judges|judging|judged|justify|justifies|justifying|justified|observe|observes|observing|observed|perceive|perceives|perceiving|perceived|predict|predicts|predicting|predicted|pretend|pretends|pretending|pretended|reckon|reckons|reckoning|reckoned|remind|reminds|reminding|reminded|satisfy|satisfies|satisfying|satisfied|solve|solves|solving|solved|study|studies|studying|studied|suspect|suspects|suspecting|suspected|trust|trusts|trusting|trusted)"

    # Facilitation or causation verbs
    vb_cause = "(help|helps|helping|helped|let|lets|letting|allow|allows|allowing|allowed|affect|affects|affecting|affected|cause|causes|causing|caused|enable|enables|enabling|enabled|ensure|ensures|ensuring|ensured|force|forces|forcing|forced|prevent|prevents|preventing|prevented|assist|assists|assisting|assisted|guarantee|guarantees|guaranteeing|guaranteed|influence|influences|influencing|influenced|permit|permits|permitting|permitted|require|requires|requiring|required)"

    # Occurrence verbs
    vb_occur = "(become|becomes|becoming|became|happen|happens|happening|happened|change|changes|changing|changed|die|dies|dying|died|grow|grows|grew|grown|growing|develop|develops|developing|developed|arise|arises|arising|arose|arisen|emerge|emerges|emerging|emerged|fall|falls|falling|fell|fallen|increase|increases|increasing|increased|last|lasts|lasting|lasted|rise|rises|rising|rose|risen|disappear|disappears|disappearing|disappeared|flow|flows|flowing|flowed|shine|shines|shining|shone|shined|sink|sinks|sank|sunk|sunken|sinking|slip|slips|slipping|slipped|occur|occurs|occurring|occurred)"

    # Existence or relationship verbs ELF: Does not include the copular BE as in Biber (2006). LOOK was also removed due to too high polysemy. 
    vb_exist =	"(seem|seems|seeming|seemed|stand|stands|standing|stood|stay|stays|staid|stayed|staying|live|lives|living|lived|appear|appears|appearing|appeared|include|includes|including|included|involve|involves|involving|involved|contain|contains|containing|contained|exist|exists|existing|existed|indicate|indicates|indicating|indicated|concern|concerns|concerning|concerned|constitute|constitutes|constituting|constituted|define|defines|defining|defined|derive|derives|deriving|derived|illustrate|illustrates|illustrating|illustrated|imply|implies|implying|implied|lack|lacks|lacking|lacked|owe|owes|owing|owed|own|owns|owning|owned|possess|possesses|possessing|possessed|suit|suits|suiting|suited|vary|varies|varying|varied|fit|fits|fitting|fitted|matter|matters|mattering|mattered|reflect|reflects|reflecting|reflected|relate|relates|relating|related|remain|remains|remaining|remained|reveal|reveals|revealing|revealed|sound|sounds|sounding|sounded|tend|tends|tending|tended|represent|represents|representing|represented|deserve|deserves|deserving|deserved)"

    # Aspectual verbs
    vb_aspect =	"(start|starts|starting|started|keep|keeps|keeping|kept|stop|stops|stopping|stopped|begin|begins|beginning|began|begun|complete|completes|completing|completed|end|ends|ending|ended|finish|finishes|finishing|finished|cease|ceases|ceasing|ceased|continue|continues|continuing|continued)"

    # Shakir: noun, adj, adv semantic categories from Biber 2006
    nn_human = "(family|families|guy|guys|individual|individuals|kid|kids|man|men|manager|managers|member|members|parent|parents|teacher|teachers|child|children|people|peoples|person|people|student|students|woman|women|animal|animals|applicant|applicants|author|authors|baby|babies|boy|boys|client|clients|consumer|consumers|critic|critics|customer|customers|doctor|doctors|employee|employees|employer|employers|father|fathers|female|females|friend|friends|girl|girls|god|gods|historian|historians|husband|husbands|American|Americans|Indian|Indians|instructor|instructors|king|kings|leader|leaders|male|males|mother|mothers|owner|owners|president|presidents|professor|professors|researcher|researchers|scholar|scholars|speaker|speakers|species|supplier|suppliers|undergraduate|undergraduates|user|users|wife|wives|worker|workers|writer|writers|accountant|accountants|adult|adults|adviser|advisers|agent|agents|aide|aides|ancestor|ancestors|anthropologist|anthropologists|archaeologist|archaeologists|artist|artists|artiste|artistes|assistant|assistants|associate|associates|attorney|attorneys|audience|audiences|auditor|auditors|bachelor|bachelors|bird|birds|boss|bosses|brother|brothers|buddha|buddhas|buyer|buyers|candidate|candidates|cat|cats|citizen|citizens|colleague|colleagues|collector|collectors|competitor|competitors|counselor|counselors|daughter|daughters|deer|defendant|defendants|designer|designers|developer|developers|director|directors|driver|drivers|economist|economists|engineer|engineers|executive|executives|expert|experts|farmer|farmers|feminist|feminists|freshman|freshmen|ecologist|ecologists|hero|heroes|host|hosts|hunter|hunters|immigrant|immigrants|infant|infants|investor|investors|jew|jews|judge|judges|lady|ladies|lawyer|lawyers|learner|learners|listener|listeners|maker|makers|manufacturer|manufacturers|miller|millers|minister|ministers|mom|moms|monitor|monitors|monkey|monkeys|neighbor|neighbors|neighbour|neighbours|observer|observers|officer|officers|official|officials|participant|participants|partner|partners|patient|patients|personnel|personnels|peer|peers|physician|physicians|plaintiff|plaintiffs|player|players|poet|poets|police|polices|processor|processors|professional|professionals|provider|providers|psychologist|psychologists|resident|residents|respondent|respondents|schizophrenic|schizophrenics|scientist|scientists|secretary|secretaries|server|servers|shareholder|shareholders|sikh|sikhs|sister|sisters|slave|slaves|son|sons|spouse|spouses|supervisor|supervisors|theorist|theorists|tourist|tourists|victim|victims|faculty|faculties|dean|deans|engineer|engineers|reader|readers|couple|couples|graduate|graduates)"
    nn_cog = "(analysis|analyses|decision|decisions|experience|experiences|assessment|assessments|calculation|calculations|conclusion|conclusions|consequence|consequences|consideration|considerations|evaluation|evaluations|examination|examinations|expectation|expectations|observation|observations|recognition|recognitions|relation|relations|understanding|understandings|hypothesis|hypotheses|ability|abilities|assumption|assumptions|attention|attentions|attitude|attitudes|belief|beliefs|concentration|concentrations|concern|concerns|consciousness|consciousnesses|concept|concepts|fact|facts|idea|ideas|knowledge|knowledges|look|looks|need|needs|reason|reasons|sense|senses|view|views|theory|theories|desire|desires|emotion|emotions|feeling|feelings|judgement|judgements|memory|memories|notion|notions|opinion|opinions|perception|perceptions|perspective|perspectives|possibility|possibilities|probability|probabilities|responsibility|responsibilities|thought|thoughts)"
    nn_concrete = "(tank|tanks|stick|sticks|target|targets|strata|stratas|telephone|telephones|string|strings|telescope|telescopes|sugar|sugars|ticket|tickets|syllabus|syllabuses|tip|tips|salt|salts|tissue|tissues|screen|screens|tooth|teeth|sculpture|sculptures|sphere|spheres|seawater|seawaters|spot|spots|ship|ships|steam|steams|silica|silicas|steel|steels|slide|slides|stem|stems|snow|snows|sodium|mud|muds|solid|solids|mushroom|mushrooms|gift|gifts|muscle|muscles|glacier|glaciers|tube|tubes|gun|guns|nail|nails|handbook|handbooks|newspaper|newspapers|handout|handouts|node|nodes|instrument|instruments|notice|notices|knot|knots|novel|novels|lava|lavas|page|pages|food|foods|transcript|transcripts|leg|legs|eye|eyes|lemon|lemons|brain|brains|magazine|magazines|device|devices|magnet|magnets|oak|oaks|manual|manuals|package|packages|marker|markers|peak|peaks|match|matches|pen|pens|metal|metals|pencil|pencils|block|blocks|pie|pies|board|boards|pipe|pipes|heart|hearts|load|loads|paper|papers|transistor|transistors|modem|modems|book|books|mole|moles|case|cases|motor|motors|computer|computers|mound|mounds|dollar|dollars|mouth|mouths|hand|hands|movie|movies|flower|flowers|object|objects|foot|feet|table|tables|frame|frames|water|waters|vessel|vessels|arm|arms|visa|visas|bar|bars|grain|grains|bed|beds|hair|hairs|body|bodies|head|heads|box|boxes|ice|ices|car|cars|item|items|card|cards|journal|journals|chain|chains|key|keys|chair|chairs|window|windows|vehicle|vehicles|leaf|leaves|copy|copies|machine|machines|document|documents|mail|mails|door|doors|map|maps|dot|dots|phone|phones|drug|drugs|picture|pictures|truck|trucks|piece|pieces|tape|tapes|note|notes|liquid|liquids|wire|wires|equipment|equipments|wood|woods|fiber|fibers|plant|plants|fig|figs|resistor|resistors|film|films|sand|sands|file|files|score|scores|seat|seats|belt|belts|sediment|sediments|boat|boats|seed|seeds|bone|bones|soil|soils|bubble|bubbles|bud|buds|water|waters|bulb|bulbs|portrait|portraits|bulletin|bulletins|step|steps|shell|shells|stone|stones|cake|cakes|tree|trees|camera|cameras|video|videos|face|faces|wall|walls|acid|acids|alcohol|alcohols|cap|caps|aluminium|aluminiums|clay|clays|artifact|artifacts|clock|clocks|rain|rains|clothing|clothings|asteroid|asteroids|club|clubs|automobile|automobiles|comet|comets|award|awards|sheet|sheets|bag|bags|branch|branches|ball|balls|copper|coppers|banana|bananas|counter|counters|band|bands|cover|covers|wheel|wheels|crop|crops|drop|drops|crystal|crystals|basin|basins|cylinder|cylinders|bell|bells|desk|desks|dinner|dinners|pole|poles|button|buttons|pot|pots|disk|disks|pottery|potteries|drain|drains|radio|radios|drink|drinks|reactor|reactors|drawing|drawings|retina|retinas|dust|dusts|ridge|ridges|edge|edges|ring|rings|engine|engines|ripple|ripples|plate|plates|game|games|cent|cents|post|posts|envelope|envelopes|rock|rocks|filter|filters|root|roots|finger|fingers|slope|slopes|fish|fish|space|spaces|fruit|fruits|statue|statues|furniture|furnitures|textbook|textbooks|gap|gaps|tool|tools|gate|gates|train|trains|gel|gels|deposit|deposits|chart|charts|mixture|mixtures)"
    nn_technical = "(cell|cells|unit|units|gene|genes|wave|waves|ion|ions|bacteria|bacterias|electron|electrons|chromosome|chromosomes|element|elements|cloud|clouds|sample|samples|isotope|isotopes|schedule|schedules|neuron|neurons|software|softwares|nuclei|nucleus|solution|solutions|nucleus|nuclei|atom|atoms|ray|rays|margin|margins|virus|viruses|mark|marks|hydrogen|hydrogens|mineral|minerals|internet|internets|molecule|molecules|mineral|minerals|organism|organisms|message|messages|oxygen|oxygens|paragraph|paragraphs|particle|particles|sentence|sentences|play|plays|star|stars|poem|poems|thesis|theses|proton|protons|unit|units|web|webs|layer|layers|center|centers|centre|centres|matter|matters|chapter|chapters|square|squares|data|circle|circles|equation|equations|compound|compounds|exam|exams|letter|letters|bill|bills|page|pages|component|components|statement|statements|diagram|diagrams|word|words|dna|angle|angles|fire|fires|carbon|carbons|formula|formulas|graph|graphs|iron|irons|lead|leads|jury|juries|light|lights|list|lists)"
    nn_place = "(apartment|apartments|interior|interiors|bathroom|bathrooms|moon|moons|bay|bays|museum|museums|bench|benches|neighborhood|neighborhoods|neighbourhood|neighbourhoods|bookstore|bookstores|opposite|opposites|border|borders|orbit|orbits|cave|caves|orbital|orbitals|continent|continents|outside|outsides|delta|deltas|parallel|parallels|desert|deserts|passage|passages|estuary|estuaries|pool|pools|factory|factories|prison|prisons|farm|farms|restaurant|restaurants|forest|forests|sector|sectors|habitat|habitats|shaft|shafts|hell|hells|shop|shops|hemisphere|hemispheres|southwest|hill|hills|station|stations|hole|holes|territory|territories|horizon|horizons|road|roads|bottom|bottoms|store|stores|boundary|boundaries|stream|streams|building|buildings|top|tops|campus|campuses|valley|valleys|canyon|canyons|village|villages|coast|coasts|city|cities|county|counties|country|countries|court|courts|earth|earths|front|fronts|environment|environments|district|districts|field|fields|floor|floors|market|markets|lake|lakes|office|offices|land|lands|organization|organizations|lecture|lectures|place|places|left|lefts|room|rooms|library|libraries|area|areas|location|locations|class|classes|middle|middles|classroom|classrooms|mountain|mountains|ground|grounds|north|norths|hall|halls|ocean|oceans|park|parks|planet|planets|property|properties|region|regions|residence|residences|river|rivers)"
    nn_quant = "(cycle|cycles|rate|rates|date|dates|second|seconds|frequency|frequencies|section|sections|future|futures|semester|semesters|half|halves|temperature|temperatures|height|heights|today|todays|number|numbers|amount|amounts|week|weeks|age|ages|day|days|century|centuries|part|parts|energy|energies|lot|lots|heat|heats|term|terms|hour|hours|time|times|month|months|mile|miles|period|periods|moment|moments|morning|mornings|volume|volumes|per|weekend|weekends|percentage|percentages|weight|weights|portion|portions|minute|minutes|quantity|quantities|percent|percents|quarter|quarters|length|lengths|ratio|ratios|measure|measures|summer|summers|meter|meters|volt|volts|voltage|voltages)"
    nn_group = "(airline|airlines|institute|institutes|colony|colonies|bank|banks|flight|flights|church|churches|hotel|hotels|firm|firms|hospital|hospitals|household|households|college|colleges|institution|institutions|house|houses|lab|labs|laboratory|laboratories|community|communities|company|companies|government|governments|university|universities|school|schools|home|homes|congress|congresses|committee|committees)"
    nn_abstract_process = "(action|actions|activity|activities|application|applications|argument|arguments|development|developments|education|educations|effect|effects|function|functions|method|methods|research|researches|result|results|process|processes|accounting|accountings|achievement|achievements|addition|additions|administration|administrations|approach|approaches|arrangement|arrangements|assignment|assignments|competition|competitions|construction|constructions|consumption|consumptions|contribution|contributions|counseling|counselings|criticism|criticisms|definition|definitions|discrimination|discriminations|description|descriptions|discussion|discussions|distribution|distributions|division|divisions|eruption|eruptions|evolution|evolutions|exchange|exchanges|exercise|exercises|experiment|experiments|explanation|explanations|expression|expressions|formation|formations|generation|generations|graduation|graduations|management|managements|marketing|marketings|marriage|marriages|mechanism|mechanisms|meeting|meetings|operation|operations|orientation|orientations|performance|performances|practice|practices|presentation|presentations|procedure|procedures|production|productions|progress|progresses|reaction|reactions|registration|registrations|regulation|regulations|revolution|revolutions|selection|selections|session|sessions|strategy|strategies|teaching|teachings|technique|techniques|tradition|traditions|training|trainings|transition|transitions|treatment|treatments|trial|trials|act|acts|agreement|agreements|attempt|attempts|attendance|attendances|birth|births|break|breaks|claim|claims|comment|comments|comparison|comparisons|conflict|conflicts|deal|deals|death|deaths|debate|debates|demand|demands|answer|answers|control|controls|flow|flows|service|services|work|works|test|tests|use|uses|war|wars|change|changes|question|questions|study|studies|talk|talks|task|tasks|trade|trades|transfer|transfers|admission|admissions|design|designs|detail|details|dimension|dimensions|direction|directions|disorder|disorders|diversity|diversities|economy|economies|emergency|emergencies|emphasis|emphases|employment|employments|equilibrium|equilibriums|equity|equities|error|errors|expense|expenses|facility|facilities|failure|failures|fallacy|fallacies|feature|features|format|formats|freedom|freedoms|fun|funs|gender|genders|goal|goals|grammar|grammars|health|healths|heat|heats|help|helps|identity|identities|image|images|impact|impacts|importance|importances|influence|influences|input|inputs|labor|labors|leadership|leaderships|link|links|manner|manners|math|maths|matrix|matrices|meaning|meanings|music|musics|network|networks|objective|objectives|opportunity|opportunities|option|options|origin|origins|output|outputs|past|pasts|pattern|patterns|phase|phases|philosophy|philosophies|plan|plans|potential|potentials|prerequisite|prerequisites|presence|presences|principle|principles|success|successes|profile|profiles|profit|profits|proposal|proposals|psychology|psychologies|quality|qualities|quiz|quizzes|race|races|reality|realities|religion|religions|resource|resources|respect|respects|rest|rests|return|returns|risk|risks|substance|substances|scene|scenes|security|securities|series|series|set|sets|setting|settings|sex|sexes|shape|shapes|share|shares|show|shows|sign|signs|signal|signals|sort|sorts|sound|sounds|spring|springs|stage|stages|standard|standards|start|starts|stimulus|stimuli|strength|strengths|stress|stresses|style|styles|support|supports|survey|surveys|symbol|symbols|topic|topics|track|tracks|trait|traits|trouble|troubles|truth|truths|variation|variations|variety|varieties|velocity|velocities|version|versions|whole|wholes|action|actions|account|accounts|condition|conditions|culture|cultures|end|ends|factor|factors|grade|grades|interest|interests|issue|issues|job|jobs|kind|kinds|language|languages|law|laws|level|levels|life|lives|model|models|name|names|nature|natures|order|orders|policy|policies|position|positions|power|powers|pressure|pressures|relationship|relationships|requirement|requirements|role|roles|rule|rules|science|sciences|side|sides|situation|situations|skill|skills|source|sources|structure|structures|subject|subjects|type|types|information|informations|right|rights|state|states|system|systems|value|values|way|ways|address|addresses|absence|absences|advantage|advantages|aid|aids|alternative|alternatives|aspect|aspects|authority|authorities|axis|axes|background|backgrounds|balance|balances|base|bases|beginning|beginnings|benefit|benefits|bias|biases|bond|bonds|capital|capitals|care|cares|career|careers|cause|causes|characteristic|characteristics|charge|charges|check|checks|choice|choices|circuit|circuits|circumstance|circumstances|climate|climates|code|codes|color|colors|column|columns|combination|combinations|complex|complexes|connection|connections|constant|constants|constraint|constraints|contact|contacts|content|contents|contract|contracts|context|contexts|contrast|contrasts|crime|crimes|criteria|criterias|cross|crosses|current|currents|curriculum|curriculums|curve|curves|debt|debts|density|densities)"
    advl_nonfact = "(confidentially|frankly|generally|honestly|mainly|technically|truthfully|typically|reportedly|primarily|usually)"
    advl_att = "(amazingly|astonishingly|conveniently|curiously|hopefully|fortunately|importantly|ironically|rightly|sadly|surprisingly|unfortunately)"
    advl_fact = "(actually|always|certainly|definitely|indeed|inevitably|never|obviously|really|undoubtedly|nodoubt|ofcourse|infact)"
    advl_likely = "(apparently|evidently|perhaps|possibly|predictably|probably|roughly|maybe)"
    jj_size = "(big|deep|heavy|huge|long|large|little|short|small|thin|wide|narrow)"
    jj_time = "(annual|daily|early|late|new|old|recent|young|weekly|monthly)"
    jj_color = "(black|white|dark|bright|blue|brown|green|gr[ae]y|red|orange|yellow|purple|pink)"
    jj_eval = "(bad|beautiful|best|fine|good|great|lovely|nice|poor)"
    jj_relation = "(additional|average|chief|complete|different|direct|entire|external|final|following|general|initial|internal|left|main|maximum|necessary|original|particular|previous|primary|public|similar|single|standard|top|various|same)"
    jj_topic = "(chemical|commercial|environmental|human|industrial|legal|medical|mental|official|oral|phonetic|political|sexual|social|ventral|visual)"
    jj_att_other = "(afraid|amazed|(un)?aware|concerned|disappointed|encouraged|glad|happy|hopeful|pleased|shocked|surprised|worried)"
    jj_epist_other = "(apparent|certain|clear|confident|convinced|correct|evident|false|impossible|inevitable|obvious|positive|right|sure|true|well-known|doubtful|likely|possible|probable|unlikely)"
    comm_vb_other = "(say|says|saying|said|tell|tells|telling|told|call|calls|calling|called|ask|asks|asking|asked|write|writes|writing|wrote|written|talk|talks|talking|talked|speak|speaks|spoke|spoken|speaking|thank|thanks|thanking|thanked|describe|describing|describes|described|claim|claims|claiming|claimed|offer|offers|offering|offered|admit|admits|admitting|admitted|announce|announces|announcing|announced|answer|answers|answering|answered|argue|argues|arguing|argued|deny|denies|denying|denied|discuss|discusses|discussing|discussed|encourage|encourages|encouraging|encouraged|explain|explains|explaining|explained|express|expresses|expressing|expressed|insist|insists|insisting|insisted|mention|mentions|mentioning|mentioned|offer|offers|offering|offered|propose|proposes|proposing|proposed|quote|quotes|quoting|quoted|reply|replies|replying|replied|shout|shouts|shouting|shouted|sign|signs|signing|signed|sing|sings|singing|sang|sung|state|states|stating|stated|teach|teaches|teaching|taught|warn|warns|warning|warned|accuse|accuses|accusing|accused|acknowledge|acknowledges|acknowledging|acknowledged|address|addresses|addressing|addressed|advise|advises|advising|advised|appeal|appeals|appealing|appealed|assure|assures|assuring|assured|challenge|challenges|challenging|challenged|complain|complains|complaining|complained|consult|consults|consulting|consulted|convince|convinces|convincing|convinced|declare|declares|declaring|declared|demand|demands|demanding|demanded|emphasize|emphasizes|emphasizing|emphasized|emphasise|emphasises|emphasising|emphasised|excuse|excuses|excusing|excused|inform|informs|informing|informed|invite|invites|inviting|invited|persuade|persuades|persuading|persuaded|phone|phones|phoning|phoned|pray|prays|praying|prayed|promise|promises|promising|promised|question|questions|questioning|questioned|recommend|recommends|recommending|recommended|remark|remarks|remarking|remarked|respond|responds|responding|responded|specify|specifies|specifying|specified|swear|swears|swearing|swore|sworn|threaten|threatens|threatening|threatened|urge|urges|urging|urged|welcome|welcomes|welcoming|welcomed|whisper|whispers|whispering|whispered|suggest|suggests|suggesting|suggested|plead|pleads|pleaded|pleading|agree|agrees|agreed|agreeing|assert|asserts|asserting|asserted|beg|begs|begging|begged|confide|confides|confiding|confided|command|commands|commanding|commanded|disagree|disagreeing|disagrees|disagreed|object|objects|objected|objects|pledge|pledges|pledging|pledged|report|reports|reported|reporting|testify|testifies|testified|testifying|vow|vows|vowing|vowed|mean|means|meaning|meant)"
    fact_vb_other = "(concluding|conclude|concluded|concludes|demonstrates|demonstrating|demonstrated|demonstrate|determining|determines|determine|determined|discovered|discovers|discover|discovering|finds|finding|found|find|knows|known|knowing|know|knew|learn|learns|learning|learnt|means|meaning|meant|mean|notifies|notices|notice|noticed|notify|notifying|noticing|notified|observed|observes|observing|observe|proven|prove|proving|proved|proves|reali(z|s)ed|reali(z|s)es|reali(z|s)e|reali(z|s)ing|recogni(z|s)es|recogni(z|s)e|recogni(z|s)ed|recogni(z|s)ing|remembered|remember|remembers|remembering|sees|seen|saw|seeing|see|showing|shows|shown|showed|show|understand|understands|understanding|understood)"
    likely_vb_other = "(assumes|assumed|assuming|assume|believe|believing|believes|believed|doubting|doubted|doubts|doubt|gathers|gathering|gathered|gather|guessed|guess|guessing|guesses|hypothesi(z|s)ing|hypothesi(z|s)ed|hypothesi(z|s)e|hypothesi(z|s)es|imagine|imagining|imagines|imagined|predict|predicted|predicting|predicts|presupposing|presupposes|presuppose|presupposed|presumes|presuming|presumed|presume|reckon|reckoning|reckoned|reckons|seemed|seems|seem|seeming|speculated|speculate|speculating|speculates|suppose|supposes|supposing|supposed|suspected|suspect|suspects|suspecting|think|thinks|thinking|thought)"
    att_vb_other = "(agreeing|agreed|agree|agrees|anticipates|anticipated|anticipate|anticipating|complain|complained|complaining|complains|conceded|concede|concedes|conceding|ensure|expecting|expect|expects|expected|fears|feared|fear|fearing|feel|feels|feeling|felt|forgetting|forgets|forgotten|forgot|forget|hoped|hope|hopes|hoping|minding|minded|minds|mind|preferred|prefer|preferring|prefers|pretending|pretend|pretended|pretends|requiring|required|requires|require|wishes|wished|wish|wishing|worry|worrying|worries|worried)"

    # Shakir: vocabulary lists for that, wh and to clauses governed by semantic classes of verbs, nouns, adjectives
    th_vb_comm = "(say|says|saying|said|tell|tells|telling|told|call|calls|calling|called|ask|asks|asking|asked|write|writes|writing|wrote|written|talk|talks|talking|talked|speak|speaks|spoke|spoken|speaking|thank|thanks|thanking|thanked|describe|describing|describes|described|claim|claims|claiming|claimed|offer|offers|offering|offered|admit|admits|admitting|admitted|announce|announces|announcing|announced|answer|answers|answering|answered|argue|argues|arguing|argued|deny|denies|denying|denied|discuss|discusses|discussing|discussed|encourage|encourages|encouraging|encouraged|explain|explains|explaining|explained|express|expresses|expressing|expressed|insist|insists|insisting|insisted|mention|mentions|mentioning|mentioned|offer|offers|offering|offered|propose|proposes|proposing|proposed|quote|quotes|quoting|quoted|reply|replies|replying|replied|shout|shouts|shouting|shouted|sign|signs|signing|signed|sing|sings|singing|sang|sung|state|states|stating|stated|teach|teaches|teaching|taught|warn|warns|warning|warned|accuse|accuses|accusing|accused|acknowledge|acknowledges|acknowledging|acknowledged|address|addresses|addressing|addressed|advise|advises|advising|advised|appeal|appeals|appealing|appealed|assure|assures|assuring|assured|challenge|challenges|challenging|challenged|complain|complains|complaining|complained|consult|consults|consulting|consulted|convince|convinces|convincing|convinced|declare|declares|declaring|declared|demand|demands|demanding|demanded|emphasize|emphasizes|emphasizing|emphasized|emphasise|emphasises|emphasising|emphasised|excuse|excuses|excusing|excused|inform|informs|informing|informed|invite|invites|inviting|invited|persuade|persuades|persuading|persuaded|phone|phones|phoning|phoned|pray|prays|praying|prayed|promise|promises|promising|promised|question|questions|questioning|questioned|recommend|recommends|recommending|recommended|remark|remarks|remarking|remarked|respond|responds|responding|responded|specify|specifies|specifying|specified|swear|swears|swearing|swore|sworn|threaten|threatens|threatening|threatened|urge|urges|urging|urged|welcome|welcomes|welcoming|welcomed|whisper|whispers|whispering|whispered|suggest|suggests|suggesting|suggested|plead|pleads|pleaded|pleading|agree|agrees|agreed|agreeing|assert|asserts|asserting|asserted|beg|begs|begging|begged|confide|confides|confiding|confided|command|commands|commanding|commanded|disagree|disagreeing|disagrees|disagreed|object|objects|objected|objects|pledge|pledges|pledging|pledged|report|reports|reported|reporting|testify|testifies|testified|testifying|vow|vows|vowing|vowed|mean|means|meaning|meant)"
    th_vb_att = "(agreeing|agreed|agree|agrees|anticipates|anticipated|anticipate|anticipating|complain|complained|complaining|complains|conceded|concede|concedes|conceding|ensure|expecting|expect|expects|expected|fears|feared|fear|fearing|feel|feels|feeling|felt|forgetting|forgets|forgotten|forgot|forget|hoped|hope|hopes|hoping|minding|minded|minds|mind|preferred|prefer|preferring|prefers|pretending|pretend|pretended|pretends|requiring|required|requires|require|wishes|wished|wish|wishing|worry|worrying|worries|worried)"
    th_vb_fact = "(concluding|conclude|concluded|concludes|demonstrates|demonstrating|demonstrated|demonstrate|determining|determines|determine|determined|discovered|discovers|discover|discovering|finds|finding|found|find|knows|known|knowing|know|knew|learn|learns|learning|learnt|means|meaning|meant|mean|notifies|notices|notice|noticed|notify|notifying|noticing|notified|observed|observes|observing|observe|proven|prove|proving|proved|proves|reali(z|s)ed|reali(z|s)es|reali(z|s)e|reali(z|s)ing|recogni(z|s)es|recogni(z|s)e|recogni(z|s)ed|recogni(z|s)ing|remembered|remember|remembers|remembering|sees|seen|saw|seeing|see|showing|shows|shown|showed|show|understand|understands|understanding|understood)"
    th_vb_likely = "(assumes|assumed|assuming|assume|believe|believing|believes|believed|doubting|doubted|doubts|doubt|gathers|gathering|gathered|gather|guessed|guess|guessing|guesses|hypothesi(z|s)ing|hypothesi(z|s)ed|hypothesi(z|s)e|hypothesi(z|s)es|imagine|imagining|imagines|imagined|predict|predicted|predicting|predicts|presupposing|presupposes|presuppose|presupposed|presumes|presuming|presumed|presume|reckon|reckoning|reckoned|reckons|seemed|seems|seem|seeming|speculated|speculate|speculating|speculates|suppose|supposes|supposing|supposed|suspected|suspect|suspects|suspecting|think|thinks|thinking|thought)"
    to_vb_desire = "(agreeing|agreed|agree|agrees|chooses|chosen|choose|choosing|chose|decide|deciding|decided|decides|hate|hates|hating|hated|hesitated|hesitates|hesitate|hesitating|hoped|hope|hopes|hoping|intended|intend|intending|intends|likes|liked|like|liking|loving|loves|love|loved|means|meaning|meant|mean|needs|need|needing|needed|planning|plan|planned|plans|preferred|prefer|preferring|prefers|prepares|prepare|preparing|prepared|refuses|refusing|refuse|refused|wanting|want|wants|wanted|wishes|wished|wish|wishing)"
    to_vb_effort = "(allowance|allowing|allowed|allowancing|allow|allowances|allows|allowanced|attempting|attempted|attempts|attempt|enables|enabled|enabling|enable|encourages|encouraging|encouraged|encourage|fails|fail|failing|failed|help|helping|helps|helped|instructs|instructed|instruct|instructing|managing|managed|manage|manages|oblige|obligate|obliged|obligates|obliging|obligating|obliges|obligated|order|ordering|orders|ordered|permitted|permits|permit|permitting|persuaded|persuades|persuade|persuading|prompts|prompting|prompted|prompt|requiring|requisitions|requisitioning|required|requires|requisition|requisitioned|require|sought|seeking|seeks|seek|try|trying|tries|tried)"
    to_vb_prob = "(appear|appeared|appears|appearing|happens|happened|happen|happening|seemed|seems|seem|seeming|tending|tends|tended|tend)"
    to_vb_speech = "(asks|ask|asking|asked|claiming|claims|claim|claimed|invite|inviting|invited|invites|promising|promised|promise|promises|reminding|remind|reminded|reminds|requesting|request|requests|requested|saying|say|said|says|teaches|teaching|taught|teach|tell|tells|telling|told|urging|urges|urged|urge|warning|warn|warned|warns)"
    to_vb_mental = "(assumed|assumes|assume|assuming|believing|believes|believe|believed|considered|considers|consider|considering|expecting|expects|expected|expect|find|found|finding|finds|forgetting|forget|forgets|forgot|forgotten|imagine|imagined|imagining|imagines|judge|adjudicates|adjudicate|judges|judged|knowing|knows|known|know|knew|learnt|learning|learns|learn|presumes|presuming|presumed|presume|pretend|pretends|pretended|pretending|remembered|remember|remembers|remembering|supposing|suppose|supposes|supposed)"
    wh_vb_att = "(agreeing|agreed|agree|agrees|anticipates|anticipated|anticipate|anticipating|complain|complained|complaining|complains|conceded|concede|concedes|conceding|ensure|expecting|expect|expects|expected|fears|feared|fear|fearing|feel|feels|feeling|felt|forgetting|forgets|forgotten|forgot|forget|hoped|hope|hopes|hoping|minding|minded|minds|mind|preferred|prefer|preferring|prefers|pretending|pretend|pretended|pretends|requiring|required|requires|require|wishes|wished|wish|wishing|worry|worrying|worries|worried)"
    wh_vb_fact = "(concluding|conclude|concluded|concludes|demonstrates|demonstrating|demonstrated|demonstrate|determining|determines|determine|determined|discovered|discovers|discover|discovering|finds|finding|found|find|knows|known|knowing|know|knew|learn|learns|learning|learnt|means|meaning|meant|mean|notifies|notices|notice|noticed|notify|notifying|noticing|notified|observed|observes|observing|observe|proven|prove|proving|proved|proves|reali(z|s)ed|reali(z|s)es|reali(z|s)e|reali(z|s)ing|recogni(z|s)es|recogni(z|s)e|recogni(z|s)ed|recogni(z|s)ing|remembered|remember|remembers|remembering|sees|seen|saw|seeing|see|showing|shows|shown|showed|show|understand|understands|understanding|understood)"
    wh_vb_likely = "(assumes|assumed|assuming|assume|believe|believing|believes|believed|doubting|doubted|doubts|doubt|gathers|gathering|gathered|gather|guessed|guess|guessing|guesses|hypothesi(z|s)ing|hypothesi(z|s)ed|hypothesi(z|s)e|hypothesi(z|s)es|imagine|imagining|imagines|imagined|predict|predicted|predicting|predicts|presupposing|presupposes|presuppose|presupposed|presumes|presuming|presumed|presume|reckon|reckoning|reckoned|reckons|seemed|seems|seem|seeming|speculated|speculate|speculating|speculates|suppose|supposes|supposing|supposed|suspected|suspect|suspects|suspecting|think|thinks|thinking|thought)"
    wh_vb_comm = "(say|says|saying|said|tell|tells|telling|told|call|calls|calling|called|ask|asks|asking|asked|write|writes|writing|wrote|written|talk|talks|talking|talked|speak|speaks|spoke|spoken|speaking|thank|thanks|thanking|thanked|describe|describing|describes|described|claim|claims|claiming|claimed|offer|offers|offering|offered|admit|admits|admitting|admitted|announce|announces|announcing|announced|answer|answers|answering|answered|argue|argues|arguing|argued|deny|denies|denying|denied|discuss|discusses|discussing|discussed|encourage|encourages|encouraging|encouraged|explain|explains|explaining|explained|express|expresses|expressing|expressed|insist|insists|insisting|insisted|mention|mentions|mentioning|mentioned|offer|offers|offering|offered|propose|proposes|proposing|proposed|quote|quotes|quoting|quoted|reply|replies|replying|replied|shout|shouts|shouting|shouted|sign|signs|signing|signed|sing|sings|singing|sang|sung|state|states|stating|stated|teach|teaches|teaching|taught|warn|warns|warning|warned|accuse|accuses|accusing|accused|acknowledge|acknowledges|acknowledging|acknowledged|address|addresses|addressing|addressed|advise|advises|advising|advised|appeal|appeals|appealing|appealed|assure|assures|assuring|assured|challenge|challenges|challenging|challenged|complain|complains|complaining|complained|consult|consults|consulting|consulted|convince|convinces|convincing|convinced|declare|declares|declaring|declared|demand|demands|demanding|demanded|emphasize|emphasizes|emphasizing|emphasized|emphasise|emphasises|emphasising|emphasised|excuse|excuses|excusing|excused|inform|informs|informing|informed|invite|invites|inviting|invited|persuade|persuades|persuading|persuaded|phone|phones|phoning|phoned|pray|prays|praying|prayed|promise|promises|promising|promised|question|questions|questioning|questioned|recommend|recommends|recommending|recommended|remark|remarks|remarking|remarked|respond|responds|responding|responded|specify|specifies|specifying|specified|swear|swears|swearing|swore|sworn|threaten|threatens|threatening|threatened|urge|urges|urging|urged|welcome|welcomes|welcoming|welcomed|whisper|whispers|whispering|whispered|suggest|suggests|suggesting|suggested|plead|pleads|pleaded|pleading|agree|agrees|agreed|agreeing|assert|asserts|asserting|asserted|beg|begs|begging|begged|confide|confides|confiding|confided|command|commands|commanding|commanded|disagree|disagreeing|disagrees|disagreed|object|objects|objected|objects|pledge|pledges|pledging|pledged|report|reports|reported|reporting|testify|testifies|testified|testifying|vow|vows|vowing|vowed|mean|means|meaning|meant)"
    th_jj_att = "(afraid|amazed|(un)?aware|concerned|disappointed|encouraged|glad|happy|hopeful|pleased|shocked|surprised|worried)"
    th_jj_fact = "(apparent|certain|clear|confident|convinced|correct|evident|false|impossible|inevitable|obvious|positive|right|sure|true|well-known)"
    th_jj_likely = "(doubtful|likely|possible|probable|unlikely)"
    th_jj_eval = "(amazing|appropriate|conceivable|crucial|essential|fortunate|imperative|inconceivable|incredible|interesting|lucky|necessary|nice|noteworthy|odd|ridiculous|strange|surprising|unacceptable|unfortunate)"
    th_nn_nonfact = "(comment|comments|news|news|proposal|proposals|proposition|propositions|remark|remarks|report|reports|requirement|requirements)"
    th_nn_att = "(grounds|ground|hope|hopes|reason|reasons|view|views|thought|thoughts)"
    th_nn_fact = "(assertion|assertions|conclusion|conclusions|conviction|convictions|discovery|discoveries|doubt|doubts|fact|facts|knowledge|knowledges|observation|observations|principle|principles|realization|realizations|result|results|statement|statements)"
    th_nn_likely = "(assumption|assumptions|belief|beliefs|claim|claims|contention|contentions|feeling|feelings|hypothesis|hypotheses|idea|ideas|implication|implications|impression|impressions|notion|notions|opinion|opinions|possibility|possibilities|presumption|presumptions|suggestion|suggestions)"
    to_jj_certain = "(apt|certain|due|guaranteed|liable|likely|prone|unlikely|sure)"
    to_jj_able = "(anxious|(un)?able|careful|determined|eager|eligible|hesitant|inclined|obliged|prepared|ready|reluctant|(un)?willing)"
    to_jj_affect = "(afraid|ashamed|disappointed|embarrassed|glad|happy|pleased|proud|puzzled|relieved|sorry|surprised|worried)"
    to_jj_ease = "(difficult|easier|easy|hard|(im)?possible|tough)"
    to_jj_eval = "(bad|worse|(in)?appropriate|good|better|best|convenient|essential|important|interesting|necessary|nice|reasonable|silly|smart|stupid|surprising|useful|useless|unreasonable|wise|wrong)"
    to_nn_stance_all = "(agreement|agreements|decision|decisions|desire|desires|failure|failures|inclination|inclinations|intention|intentions|obligation|obligations|opportunity|opportunities|plan|plans|promise|promises|proposal|proposals|reluctance|reluctances|responsibility|responsibilities|right|rights|tendency|tendencies|threat|threats|wish|wishes|willingness|willingnesses)"
    nn_stance_pp = "(assertion|assertions|conclusion|conclusions|conviction|convictions|discovery|discoveries|doubt|doubts|fact|facts|knowledge|knowledges|observation|observations|principle|principles|realization|realizations|result|results|statement|statements|assumption|assumptions|belief|beliefs|claim|claims|contention|contentions|feeling|feelings|hypothesis|hypotheses|idea|ideas|implication|implications|impression|impressions|notion|notions|opinion|opinions|possibility|possibilities|presumption|presumptions|suggestion|suggestions|grounds|ground|hope|hopes|reason|reasons|view|views|thought|thoughts|comment|comments|news|news|proposal|proposals|proposition|propositions|remark|remarks|report|reports|requirement|requirements|agreement|agreements|decision|decisions|desire|desires|failure|failures|inclination|inclinations|intention|intentions|obligation|obligations|opportunity|opportunities|plan|plans|promise|promises|reluctance|reluctances|responsibility|responsibilities|right|rights|tendency|tendencies|threat|threats|wish|wishes|willingness|willingnesses)"

    #---------------------------------------------------
    # COMPLEX TAGS
    for j, value in enumerate(words):
        #skip if space
        if value != " ":        
            #----------------------------------------------------
            # Shakir: Add two sub classes of attributive and predicative adjectives. 
            # The predicative counterparts should not have a TO or THSC afterwards

            if (re.search("\\b(" + jj_att_other + ")_(JJAT|JJPR)", words[j], re.IGNORECASE) and not re.search("to_|_THSC", words[j+1])): 
                words[j] = re.sub("_(\w+)", "_\\1 JJATDother", words[j])

            if (re.search("\\b(" + jj_epist_other + ")_(JJAT|JJPR)", words[j], re.IGNORECASE) and not re.search("to_|_THSC", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 JJEPSTother", words[j])

            #---------------------------------------------------

            # ELF: New features (for extended MFTE python output) 
            # Tags for comparative and superlative constructions

            # Superlatives
            if ((re.search("est_J|est_RB|\\bworst_|\\bbest-", words[j], re.IGNORECASE) and not re.search("\\btest|honest_|west_|\\bpest_|\\blest_|\\bguest_", words[j], re.IGNORECASE)) or # E.g., widest, furthest, worst, best-looking
            (re.search("\\bthe_", words[j-1], re.IGNORECASE) and re.search("\\bleast_|\\bmost_", words[j]) and re.search("_J|_RB|_NN", words[j+1]))): # E.g., the most pressing, the cheapest
                words[j] = re.sub("_(\w+)", "_\\1 SUPER", words[j])

            # Comparatives
            if ((re.search("er_J|er_RB|\\bworse_|\\bbetter-", words[j], re.IGNORECASE) and not re.search("\\bafter_|never_|rather_|other_|\\bever_|either_|together_|proper_|super_|clever_|\\beager_|queer_|hyper_|\\butter_|\\binner_|bitter_|premier_|sinister_|\\bsober_|order_|over_", words[j], re.IGNORECASE)) or # E.g., wider, further, better-looking.
            (re.search("\\bless_|\\bmore_", words[j]) and re.search("_J|_RB", words[j+1]))): # E.g., more pressing, less important.
                words[j] = re.sub("_(\w+)", "_\\1 COMPAR", words[j])
        
        #----------------------------------------------------
        # Shakir: TO and split infinitive clauses followed by vb, adj and nouns.

            if ((re.search("\\b(" + to_vb_desire + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_desire + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+2])) or
            (re.search("\\b(" + to_vb_desire + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_desire + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+2]))):
                words[j] = re.sub("_(\w+)", "_\\1 ToVDSR", words[j])

            if ((re.search("\\b(" + to_vb_effort + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_effort + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+2])) or
            (re.search("\\b(" + to_vb_effort + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_effort + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+2]))):
                words[j] = re.sub("_(\w+)", "_\\1 ToVEFRT", words[j])

            if ((re.search("\\b(" + to_vb_prob + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_prob + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+2])) or
            (re.search("\\b(" + to_vb_prob + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_prob + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+2]))):
                words[j] = re.sub("_(\w+)", "_\\1 ToVPROB", words[j])

            if ((re.search("\\b(" + to_vb_speech + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_speech + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+2])) or
            (re.search("\\b(" + to_vb_speech + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_speech + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+2]))):
                words[j] = re.sub("_(\w+)", "_\\1 ToVSPCH", words[j])

            if ((re.search("\\b(" + to_vb_mental + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_mental + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+2])) or
            (re.search("\\b(" + to_vb_mental + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+1])) or
            (re.search("\\b(" + to_vb_mental + ")_V", words[j-1], re.IGNORECASE) and re.search("\\bna_TO", words[j]) and re.search("\_V", words[j+2]))):
                words[j] = re.sub("_(\w+)", "_\\1 ToVMNTL", words[j])

            if (re.search("\\b(" + to_jj_certain + ")_J", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJCRTN", words[j])

            if (re.search("\\b(" + to_jj_able + ")_J", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJABL", words[j])

            if (re.search("\\b(" + to_jj_affect + ")_J", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJEFCT", words[j])

            if (re.search("\\b(" + to_jj_ease + ")_J", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJEASE", words[j])

            if (re.search("\\b(" + to_jj_eval + ")_J", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJEVAL", words[j])

            # Shakir: sums of that-clauses for vb, jj, nn and all to be used if original are too low freq
                    
            if (re.search(" (ToVDSR|ToVEFRT|ToVPROB|ToVSPCH|ToVMNTL)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ToVSTNCall", words[j])

            if (re.search(" (ToJCRTN|ToJABL|ToJEFCT|ToJEASE|ToJEVAL)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ToJSTNCall", words[j])

        # # Shakir: all to vb stance excep verbs of desive which are frequent mainly due to want to constructions
        # if (re.search(" (ToVEFRT|ToVPROB|ToVSPCH|ToVMNTL)", words[j])):
        #     words[j] = re.sub("_(\w+)", "_\\1 ToVSTNCother", words[j])

            if (re.search("\\b(" + to_nn_stance_all + ")_N", words[j-1], re.IGNORECASE) and re.search("\\bto_", words[j]) and re.search("\_V", words[j+1])):
                words[j] = re.sub("_(\w+)", "_\\1 ToNSTNC", words[j])

            if (re.search(" (ToVDSR|ToVEFRT|ToVPROB|ToVSPCH|ToVMNTL|ToJCRTN|ToJABL|ToJEFCT|ToJEASE|ToJEVAL|ToNSTNC)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ToSTNCall", words[j])

            #---------------------------------------------------
            # Shakir: That complement clauses as tagged previously by THSC
            if (re.search("\\b(" + th_vb_comm + ")_V", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThVCOMM", words[j])

            if (re.search("\\b(" + th_vb_att + ")_V", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThVATT", words[j])

            if (re.search("\\b(" + th_vb_fact + ")_V", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThVFCT", words[j])

            if (re.search("\\b(" + th_vb_likely + ")_V", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThVLIK", words[j])

            if (re.search("\\b(" + th_jj_att + ")_J", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThJATT", words[j])

            if (re.search("\\b(" + th_jj_fact + ")_J", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThJFCT", words[j])

            if (re.search("\\b(" + th_jj_likely + ")_J", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThJLIK", words[j])

            if (re.search("\\b(" + th_jj_eval + ")_J", words[j-1], re.IGNORECASE) and re.search("_THSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThJEVL", words[j])

            # Shakir: that relative clauses related to attitude
            if (re.search("\\b(" + th_nn_nonfact + ")_N", words[j-1], re.IGNORECASE) and re.search("_THRC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThNNFCT", words[j])

            if (re.search("\\b(" + th_nn_att + ")_N", words[j-1], re.IGNORECASE) and re.search("_THRC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThNATT", words[j])

            if (re.search("\\b(" + th_nn_fact + ")_N", words[j-1], re.IGNORECASE) and re.search("_THRC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThNFCT", words[j])

            if (re.search("\\b(" + th_nn_likely + ")_N", words[j-1], re.IGNORECASE) and re.search("_THRC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThNLIK", words[j])

            # Shakir: wh sub clauses after verb classes
            if (re.search("\\b(" + wh_vb_att + ")_V", words[j-1], re.IGNORECASE) and re.search("_WHSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WhVATT", words[j])

            if (re.search("\\b(" + wh_vb_fact + ")_V", words[j-1], re.IGNORECASE) and re.search("_WHSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WhVFCT", words[j])

            if (re.search("\\b(" + wh_vb_likely + ")_V", words[j-1], re.IGNORECASE) and re.search("_WHSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WhVLIK", words[j])

            if (re.search("\\b(" + wh_vb_comm + ")_V", words[j-1], re.IGNORECASE) and re.search("_WHSC", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WhVCOM", words[j])

            # Shakir: preposition after stance nouns
            if (re.search("\\b(" + nn_stance_pp + ")_N", words[j-1], re.IGNORECASE) and re.search("_IN", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 PrepNSTNC", words[j])

    #-------------------------------------------------- 
    # Shakir: Nouns semantic classes (from Biber 2006)
    for index, x in enumerate(words):
        
        if x != " ":
            #--------------------------------------------------------------  
            # Shakir: noun and adverb semantic categories from Biber 2006, if there is no additional tag added previously (hence the space check)
            if (re.search("\\b(" + nn_human + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNHUMAN", words[index])
            
            if (re.search("\\b(" + nn_cog + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNCOG", words[index])

            if (re.search("\\b(" + nn_concrete + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNCONC", words[index])

            if (re.search("\\b(" + nn_place + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNPLACE", words[index])

            if (re.search("\\b(" + nn_quant + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNQUANT", words[index])

            if (re.search("\\b(" + nn_group + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNGRP", words[index])

            if (re.search("\\b(" + nn_technical + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNTECH", words[index])

            if (re.search("\\b(" + nn_abstract_process + ")_N", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 NNABSPROC", words[index])

            if (re.search("\\b(" + jj_size + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJSIZE", words[index])

            if (re.search("\\b(" + jj_time + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJTIME", words[index])

            if (re.search("\\b(" + jj_color + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJCOLR", words[index])

            if (re.search("\\b(" + jj_eval + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJEVAL", words[index])

            if (re.search("\\b(" + jj_relation + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJREL", words[index])

            if (re.search("\\b(" + jj_topic + ")_J", words[index], re.IGNORECASE) and not re.search(" ", words[index])):
                words[index] = re.sub("_(\w+)", "_\\1 JJTOPIC", words[index])

            # ELF: tags activity verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or_PASS.
            if (re.search("\\b(" + vb_act + ")_V|\\b(" + vb_act + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 ACT", words[index])

            # ELF: tags communication verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_comm + ")_V|\\b(" + vb_comm + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 COMM", words[index])

            # ELF: tags mental verbs (including the "no" in "I dunno" and "wa" in wanna). 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_mental + ")_V|\\b(" + vb_mental + ")_P|\\bno_VB", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 MENTAL", words[index])
        
            # ELF: tags causative verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_cause + ")_V|\\b(" + vb_cause + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 CAUSE", words[index])

            # ELF: tags occur verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_occur + ")_V|\\b(" + vb_occur + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 OCCUR", words[index])

            # ELF: tags existential verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_exist + ")_V|\\b(" + vb_exist + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 EXIST", words[index])

            # ELF: tags aspectual verbs. 
            # Note that adding _P is important to capture verbs tagged as PEAS, PROG or PASS.
            if (re.search("\\b(" + vb_aspect + ")_V|\\b(" + vb_aspect + ")_P", words[index], re.IGNORECASE)):
                words[index] = re.sub("_(\w+)", "_\\1 ASPECT", words[index])

            #--------------------------------------------------------------  

    # Shakir: additional semantic categories
    for j, value in enumerate(words):
        #skip if space
        if value != " ":        
            #---------------------------------------------------

            # Shakir: Nini's (2014) implementation for nominalisations with a length check more than 5 characters, and no space means no other extra tag added
            if (re.search("tions?_NN|ments?_NN|ness_NN|nesses_NN|ity_NN|ities_NN", words[j], re.IGNORECASE) and re.search("[a-z]{5,}", words[j], re.IGNORECASE) and not re.search(" ", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 NOMZ", words[j])

            # Shakir: Semantic classes of adverbs
            if ((re.search("\\b" + advl_att + "_R", words[j], re.IGNORECASE) and not re.search(" ", words[j])) or
            (re.search("\\b(even)_R", words[j], re.IGNORECASE) and re.search("\\b(worse)_", words[j+1], re.IGNORECASE) and not re.search(" ", words[j]))):
                words[j] = re.sub("_(\w+)", "_\\1 RATT", words[j])

            if (re.search("\\b(" + advl_nonfact + ")_R", words[j], re.IGNORECASE) and not re.search(" ", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 RNONFACT", words[j])

            if ((re.search("\\b(" + advl_fact + ")_R", words[j], re.IGNORECASE) and not re.search(" ", words[j])) or
            (re.search("\\b(of)_", words[j-1], re.IGNORECASE) and re.search("\\b(course)_", words[j], re.IGNORECASE)) or
            (re.search("\\b(in)_", words[j-1], re.IGNORECASE) and re.search("\\b(fact)_", words[j], re.IGNORECASE)) or
            (re.search("\\b(without|no)_", words[j-1], re.IGNORECASE) and re.search("\\b(doubt)_", words[j], re.IGNORECASE))):
                words[j] = re.sub("_(\w+)", "_\\1 RFACT", words[j])

            # Shakir: stance nouns without prep
            # check for no doubt and without doubt which are already tagged as factive adverb phrases 
            if (re.search("\\b(" + nn_stance_pp + ")_N", words[j], re.IGNORECASE) and not re.search("_IN", words[j+1]) and not re.search(" (RFACT|HDG)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 NSTNCother", words[j])

            # Shakir: Added new variable to avoid overlap in the above two sub classes and JJAT/JJPR
            if (re.search("_JJAT$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 JJATother", words[j])

            if (re.search("_JJPR$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 JJPRother", words[j])

            # Shakir: Added new variable to avoid overlap in THSC and all above TH_J and TH_V clauses
            if (re.search("_THSC$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 THSCother", words[j])

            # Shakir: Added new variable to avoid overlap in THRC and all above TH_N clauses
            if (re.search("_THRC$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 THRCother", words[j])

            # Shakir: Added new variable to avoid overlap in WHSC and WH_V clauses
            if (re.search("_WHSC$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WHSCother", words[j])

            # Shakir: Added new variable to avoid overlap in NN and N semantic/other sub classes
            if (re.search("_NN$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 NNother", words[j])

            # Shakir: Added new variable to avoid overlap in RB and R semantic sub classes
            if (re.search("_RB$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 RBother", words[j])

            # Shakir: Added new variable to avoid overlap in IN and PrepNNStance
            if (re.search("_IN$", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 INother", words[j])

            # Shakir: commented due to overlap with MENTAL and COMM verbs. They are counted with that and to phrases so no need to add any additional category
            # # Shakir: verbs in contexts other than _WHSC, _THSC or to_ . Additionally not assigned to another tag.
            # if (re.search("\\b(" + comm_vb_other + ")_V", words[j], re.IGNORECASE) and not re.search("_WHSC|_THSC|to_", words[j+1]) and not re.search(" ", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 VCOMMother", words[j])
        
            # if (re.search("\\b(" + att_vb_other + ")_V", words[j], re.IGNORECASE) and not re.search("_WHSC|_THSC|to_", words[j+1]) and not re.search(" ", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 VATTother", words[j])
        
            # if (re.search("\\b(" + fact_vb_other + ")_V", words[j], re.IGNORECASE) and not re.search("_WHSC|_THSC|to_", words[j+1]) and not re.search(" ", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 VFCTother", words[j])
        
            # if (re.search("\\b(" + likely_vb_other + ")_V", words[j], re.IGNORECASE) and not re.search("_WHSC|_THSC|to_", words[j+1]) and not re.search(" ", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 VLIKother", words[j])

            # Shakir: sums of that clauses for vb, jj, nn and all to be used if original are too low freq
            if (re.search(" (ThVCOMM|ThVATT|ThVFCT|ThVLIK)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThVSTNCall", words[j])

            # # Shakir: sums of that clauses for vb other than comm verbs
            # if (re.search(" (ThVATT|ThVFCT|ThVLIK)", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 ThVSTNCother", words[j])

            if (re.search(" (ThJATT|ThJFCT|ThJLIK|ThJEVL)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThJSTNCall", words[j])

            if (re.search(" (ThNNFCT|ThNATT|ThNFCT|ThNLIK)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThNSTNCall", words[j])

            if (re.search(" (ThVCOMM|ThVATT|ThVFCT|ThVLIK|ThJATT|ThJFCT|ThJLIK|ThJEVL|ThNNFCT|ThNATT|ThNFCT|ThNLIK)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ThSTNCall", words[j])

            # Shakir: wh vb stance all
            if (re.search(" (WhVATT|WhVFCT|WhVLIK|WhVCOM)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 WhVSTNCall", words[j])

            # Shakir: adverb stance all
            if (re.search(" (RATT|RNONFACT|RFACT|RLIKELY)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 RSTNCall", words[j])

            # # Shakir: adverb stance other than RFACT
            # if (re.search(" (RATT|RNONFACT|RLIKELY)", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 RSTNCother", words[j])

            # Shakir: all possibility modals as in Biber 1988
            if (re.search("(MDCA|MDCO|MDMM)", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 MDPOSSCall", words[j])

            # Shakir: all prediction modals as in Biber 1988 + "Going to + infinitive"
            if (re.search("(MDWS|MDWO|GTO)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 MDPREDall", words[j])

            # Shakir: all passive voice (sum of PASS and PGET)
            if (re.search("(PASS|PGET)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 PASSall", words[j])

            # Shakir: all stance noun complements (To + Th)
            if (re.search("(ToNSTNC|ThNSTNCall)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 ToThNSTNCall", words[j])

            # # Shakir: consolidate description adjectives
            # if (re.search("(JJSIZE|JJCOLR)\\b", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 JJDESCall", words[j])

            # # Shakir: consolidate stance adjectives
            # if (re.search("(JJEPSTother|JJATDother)\\b", words[j])):
            #     words[j] = re.sub("_(\w+)", "_\\1 JJEpstAtdOther", words[j])

            # Shakir: All 1st person references in one tag (equivalent of FPP1 in Biber 1988)
            if (re.search("_(PP1P|PP1S)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 PP1all", words[j])

            # Shakir: All 3rd person references to 1 tag (equivalent of TPP3 in Biber 1988)
            if (re.search("_(PP3t|PP3f|PP3m)\\b", words[j])):
                words[j] = re.sub("_(\w+)", "_\\1 PP3all", words[j])

    return words

def run_process_sentence(file: str, extended: bool = True) -> list:
    """Returns list of words after running process_sentence on it
    Args:
        file (str): text file path that is to be opened
        extended (bool): If extended semantic categories should be tagged
    Returns:
        words_tagged (list): list of words after MD tagging
    """
    text = open(file=file, encoding='utf-8', errors='ignore').read()
    sentences = re.split("[\r\n]+", text) # split on new line or carriage return to get sentences
    sentences = [re.split(' ', s) for s in sentences] # split ind. sentences on space (now it is a list of lists)
    sentences_with_buffer_spaces = ([' '] * 20)
    for sentence in sentences:
        # add a buffer of 20 empty strings to avoid IndexError which will break the loop and cause below if conditions not to be applied in process_sentence
        sentences_with_buffer_spaces = sentences_with_buffer_spaces + sentence + ([' '] * 20)
    words_tagged = process_sentence(sentences_with_buffer_spaces, extended)
    if extended:
        words_tagged = process_sentence_extended(words_tagged)
    words_tagged = [word for word in words_tagged if word != " "] # remove white space elements added prior to process_sentence
    return words_tagged


def process_file (file_dir_pair: tuple) -> None:
    """Read a given file, tag it through process_sentence and write it
    Args:
        file_dir_pair (tuple): first element is the file, second element output_dir
        extended (bool): If extended semantic categories should be tagged
    """
    file = file_dir_pair[0]
    output_dir = file_dir_pair[1]
    extended = file_dir_pair[2]
    file_name = os.path.basename(file)
    words_tagged = run_process_sentence(file, extended)
    with open(file=output_dir+file_name, mode='w', encoding='UTF-8') as f:
        f.write("\n".join(words_tagged).strip())
    return  "MD tagger tagged: " + file

def tag_MD_parallel (input_dir: str, output_dir: str, extended: bool = True) -> None:
    """Tags Stanford Tagger output files and writes in a directory names MD
    Args:
        input_dir (str): dir with Stanford Tagger tagged files
        output_dir (str): dir to write MD tagged files
        extended (bool): If extended semantic categories should be tagged
    """
    # check if dir exists, otherwise make one
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = glob.glob(input_dir + "*.txt")
    file_with_dir = [(file, output_dir, extended) for file in files]
    cpu_count = int(multiprocessing.cpu_count() / 2) #run half cpus
    with multiprocessing.Pool(cpu_count) as pool:
	    # call the function for each item in parallel
        results = list(tqdm.tqdm(pool.map(process_file, file_with_dir), total=len(file_with_dir)))
        pool.close()
        pool.join()
        for s in results:
            print(s)

def tag_MD (input_dir: str, output_dir: str, extended: bool = True) -> None:
    """Tags Stanford Tagger output files and writes in a directory names MD
    Args:
        input_dir (str): dir with Stanford Tagger tagged files
        output_dir (str): dir to write MD tagged files
        extended (bool): If extended semantic categories should be tagged
    """
    # check if dir exists, otherwise make one
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = glob.glob(input_dir + "*.txt")
    for file in files:
        print("MD tagger tagging:", file)
        file_name = os.path.basename(file)
        words_tagged = run_process_sentence(file, extended)
        with open(file=output_dir+file_name, mode='w', encoding='UTF-8') as f:
            f.write("\n".join(words_tagged).strip())
        # break

def get_ttr(tokens: list, n: int) -> float:
    """Retuns type token ratio based on the first n words as specified in user input number of tokens n
    Args:
        tokens (list): list of tokens to count TTR
        n (int): number of tokens to consider (should be at least as long as the shortest text in the corpus!)
    Returns:
        tt_ratio (float): Type Token Ratio
    """
    if len(tokens) >= n: # if len tokens greater than or equal to n
        # take first n tokens from tokens list
        temp_tokens = tokens[:n]
        # Shakir: use tokens list passed from below, split word_TAG, keep word, convert to lower case, make a list, convert to set (i.e. unique values only), find length
        n_types = len(set([word.split('_')[0].lower() for word in temp_tokens]))
        tt_ratio = n_types/n
    else: # otherwise use the whole tokens
        # Shakir: use tokens list passed from below, split word_TAG, keep word, convert to lower case, make a list, convert to set (i.e. unique values only), find length
        n_types = len(set([word.split('_')[0].lower() for word in tokens]))
        tt_ratio = n_types/len(tokens)
    return tt_ratio

def get_complex_normed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Returns raw counts df after normalizing per 100 nouns and verbs, and other words
    Args:
        df (pd.DataFrame): df of raw counts
    Returns:
        pd.DataFrame: new df with normed ounts
    """
    df_new: pd.DataFrame  = df.copy(deep=True)
    # multiply by 100
    cols_without_averages = [col for col in df_new.columns if col not in ["Filename", "Words", "AWL", "TTR", "LDE", "Ntotal", "VBtotal"]]
    df_new.loc[:, cols_without_averages] = df_new.loc[:, cols_without_averages].mul(100) #multiply by 100
    # List of features to be normalised per 100 nouns:
    # Shakir: noun semantic classes will be normalized per 100 nouns "NNHUMAN", "NNCOG", "NNCONC", "NNTECH", "NNPLACE", "NNQUANT", "NNGRP", "NNTECH", "NNABSPROC", "NOMZ", "NSTNCother"
    # Shakir: noun governed that clauses will be normalized per 100 nouns "ThNNFCT", "ThNATT", "ThNFCT", "ThNLIK", "ToNSTNC", "ToThNSTNCall", "PrepNSTNC". THRCother is THRC minus TH_N clauses
    # Shakir: two sub classes of attributive adjectives "JJEPSTother", "JJATDother", also dependent on nouns. "JJATother" is JJAT minus the prev two classes
    # Shakir: STNCall variables combine stance related sub class th and to clauses, either use individual or All counterparts "ThNSTNCall"
    NNTnorm = ["DT", "JJAT", "POS", "NCOMP", "QUAN", "NNHUMAN", "NNCOG", "NNCONC", "NNTECH", "NNPLACE", "NNQUANT", "NNGRP", "NNABSPROC", "ThNNFCT", "ThNATT", "ThNFCT", "ThNLIK", "JJEPSTother", "JJATDother", "ToNSTNC", "PrepNSTNC", "JJATother", "ThNSTNCall", "NOMZ", "NSTNCother", "JJDESCall", "JJEpstAtdOther", "JJSIZE", "JJTIME", "JJCOLR", "JJEVAL", "JJREL", "JJTOPIC", "JJSTNCallother", "NNMention", "NNP"]
    NNTnorm = [nn for nn in NNTnorm if nn in df_new.columns] #make sure every feature exists in df column
    df_new.loc[:, NNTnorm] = df_new.loc[:, NNTnorm].div(df_new.Ntotal.values, axis=0) #divide by total nouns (noun-based normalisation)
    # Features to be normalised per 100 (very crudely defined) finite verbs:
    # Shakir: vb complement clauses of various sorts will be normalized per 100 verbs "ThVCOMM", "ThVATT", "ThVFCT", "ThVLIK", "WhVATT", "WhVFCT", "WhVLIK", "WhVCOM", "ToVDSR", "ToVEFRT", "ToVPROB", "ToVSPCH", "ToVMNTL", "VCOMMother", "VATTother", "VFCTother", "VLIKother"
    # Shakir: th jj clauses are verb gen verb dependant (pred adj) so "ThJATT", "ThJFCT", "ThJLIK", "ThJEVL", will be normalized per 100 verbs
    # Shakir: note THSCother and WHSCother are THSC and WHSC minus all new above TH and WH verb/adj clauses, "JJPRother" is JJPR without epistemic and attitudinal adjectives
    # Shakir: STNCall variables combine stance related sub class th and to clauses, either use individual or All counterparts "ToVSTNCall", "ToVSTNCother", "ThVSTNCall", "ThVSTNCother", "ThJSTNCall"
    FVnorm = ["ACT", "ASPECT", "CAUSE", "COMM", "CUZ", "CC", "CONC", "COND", "EX", "EXIST", "ELAB", "FREQ", "JJPR", "MENTAL", "OCCUR", "DOAUX", "QUTAG", "QUPR", "SPLIT", "STPR", "WHQU", "THSC", "WHSC", "CONT", "VBD", "VPRT", "PLACE", "PROG", "HGOT", "BEMA", "MDCA", "MDCO", "TIME", "THATD", "THRC", "VIMP", "MDMM", "ABLE", "MDNE", \
        "MDWS", "MDWO", "XX0", "PASS", "PGET", "VBG", "VBN", "PEAS", "GTO", "PP1S", "PP1P", "PP3f", "PP3m", "PP3t", "PP2", "PIT", "PRP", "RP", "ThVCOMM", "ThVATT", "ThVFCT", "ThVLIK", "WhVATT", "WhVFCT", "WhVLIK", "WhVCOM", "ToVDSR", "ToVEFRT", "ToVPROB", "ToVSPCH", "ToVMNTL", "JJPRother", "VCOMMother", "VATTother", "VFCTother", \
            "VLIKother", "ToVSTNCall", "ThVSTNCall", "ThJSTNCall", "ThJATT", "ThJFCT", "ThJLIK", "ThJEVL", "ToVSTNCother", "PP1all", "PP3all", "WHSCother", "THSCother", "THRCother", "MDPOSSCall", "MDPREDall", "PASSall", "WhVSTNCall", "MDother", "PRPother"]
    FVnorm = [vb for vb in FVnorm if vb in df_new.columns] #make sure every feature exists in df column
    df_new.loc[:, FVnorm] = df_new.loc[:, FVnorm].div(df_new.VBtotal.values, axis=0)#.fillna(0) #divide by total verbs (finite verb phrase-based normalisation)
    # All other features should be normalised per 100 words:
    other_cols = [col for col in df_new.columns if col not in NNTnorm if col not in FVnorm] #remove nouns and verbs related cols
    other_cols = [col for col in other_cols if col not in ["Filename", "Words", "AWL", "TTR", "LDE", "Ntotal", "VBtotal"]] # exclude total counts and averages
    df_new.loc[:, other_cols] = df_new.loc[:, other_cols].div(df_new.Words.values, axis=0) #divide by total words (word-based normalisation)
    return df_new.fillna(0)

def get_wordbased_normed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Returns raw counts df after normalizing per 100 words
    Args:
        df (pd.DataFrame): df of raw counts
    Returns:
        pd.DataFrame: new df with normed ounts
    """
    df_new: pd.DataFrame  = df.copy(deep=True)
    # multiply by 100
    cols_without_averages = [col for col in df_new.columns if col not in ["Filename", "Words", "AWL", "TTR", "LDE", "Ntotal", "VBtotal"]]
    df_new.loc[:, cols_without_averages] = df_new.loc[:, cols_without_averages].mul(100) # multiply by 100
    df_new.loc[:, cols_without_averages] = df_new.loc[:, cols_without_averages].div(df.Words.values, axis=0) # divide by total number of words
    return df_new

def sort_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Retuns df columns sorted alphabatically first simple then extended tags

    Args:
        df (pd.DataFrame): df to be sorted

    Returns:
        df_sorted (pd.DataFrame): sorted df
    """
    non_tag = [col for col in df.columns if col in ["Filename", "Words", "AWL", "TTR", "LDE"]]
    simple = [col for col in df.columns if col in ["ABLE", "ACT", "AMP", "ASPECT", "BEMA", "CAUSE", "CC", "CD", "COMM", "CONC", "COND", "CONT", "CUZ", "DEMO", "DMA", "DOAUX", "DT", "DWNT", "ELAB", "EMO", "EMPH", "EX", "EXIST", "FPUH", "FREQ", "GTO", "HDG", "HGOT", "HST", "IN", "JJAT", "JJPR", "LIKE", "MDCA", "MDCO", "MDMM", "MDNE", "MDWO", "MDWS", "MENTAL", "NCOMP", "NN", "OCCUR", "PASS", "PEAS", "PGET", "PIT", "PLACE", "POLITE", "POS", "PP1P", "PP1S", "PP2", "PP3f", "PP3m", "PP3t", "PPother", "PROG", "QUAN", "QUPR", "QUTAG", "RB", "RP", "SO", "SPLIT", "STPR", "THATD", "THRC", "THSC", "TIME", "URL", "VBD", "VBG", "VBN", "VIMP", "VPRT", "WHQU", "WHSC", "XX0", "YNQU", "Ntotal", "VBtotal"]]
    simple.sort()
    extended = [col for col in df.columns if col in ["COMPAR", "INother", "JJATDother", "JJATother", "JJCOLR", "JJEPSTother", "JJEVAL", "JJPRother", "JJREL", "JJSIZE", "JJTIME", "JJTOPIC", "MDPOSSCall", "MDPREDall", "NNABSPROC", "NNCOG", "NNCONC", "NNGRP", "NNHUMAN", "NNother", "NNP", "NNPLACE", "NNQUANT", "NNTECH", "NOMZ", "NSTNCother", "PASSall", "PP1all", "PP3all", "PrepNSTNC", "RATT", "RBother", "RFACT", "RLIKELY", "RNONFACT", "RSTNCall", "SUPER", "ThJATT", "ThJEVL", "ThJFCT", "ThJLIK", "ThJSTNCall", "ThNATT", "ThNFCT", "ThNLIK", "ThNNFCT", "ThNSTNCall", "THRCother", "THSCother", "ThSTNCall", "ThVATT", "ThVCOMM", "ThVFCT", "ThVLIK", "ThVSTNCall", "ToJABL", "ToJCRTN", "ToJEASE", "ToJEFCT", "ToJEVAL", "ToJSTNCall", "ToNSTNC", "ToSTNCall", "ToVDSR", "ToVEFRT", "ToVMNTL", "ToVPROB", "ToVSPCH", "ToVSTNCall", "VATTother", "VCOMMother", "VFCTother", "VLIKother", "WHSCother", "WhVATT", "WhVCOM", "WhVFCT", "WhVLIK", "WhVSTNCall"]]
    extended.sort()
    df_simple = df[simple].reindex(columns=simple)
    df_extended = df[extended].reindex(columns=extended)
    df_non_tag = df[non_tag].reindex(columns=non_tag)
    df_sorted = pd.concat([df_non_tag, df_simple, df_extended], axis=1)
    return df_sorted


def do_counts(dir_in: str, dir_out: str, n_tokens: int) -> None:
    """Read files and count tags added by process_sentence
    Args:
        input_dir (str): dir where MD tagged files are
        output_dir (str): dir where statistics files to be created
        ttr (int): number of tokens to consider as given by the user
    """
    features_to_be_removed_from_final_table = ['NFP', 'GW', 'HYPH', 'ADD', 'AFX', 'VB', 'LIKE', 'SO', 'PPother']
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    function_words_re = "(a|about|above|after|again|ago|ai|all|almost|along|already|also|although|always|am|among|an|and|another|any|anybody|anything|anywhere|are|are|around|as|at|back|be|been|before|being|below|beneath|beside|between|beyond|billion|billionth|both|but|by|can|can|could|cos|cuz|did|do|does|doing|done|down|during|each|eight|eighteen|eighteenth|eighth|eightieth|eighty|either|eleven|eleventh|else|enough|even|ever|every|everybody|everyone|everything|everywhere|except|far|few|fewer|fifteen|fifteenth|fifth|fiftieth|fifty|first|five|for|fortieth|forty|four|fourteen|fourteenth|fourth|from|get|gets|getting|got|had|has|have|having|he|hence|her|here|hers|herself|him|himself|his|hither|how|however|hundred|hundredth|i|if|in|into|is|it|its|itself|just|last|less|many|may|me|might|million|millionth|mine|more|most|much|must|my|myself|near|near|nearby|nearly|neither|never|next|nine|nineteen|nineteenth|ninetieth|ninety|ninth|no|nobody|none|noone|nor|not|nothing|now|nowhere|of|off|often|on|once|one|only|or|other|others|ought|our|ours|ourselves|out|over|quite|rather|round|second|seven|seventeen|seventeenth|seventh|seventieth|seventy|shall|sha|she|should|since|six|sixteen|sixteenth|sixth|sixtieth|sixty|so|some|somebody|someone|something|sometimes|somewhere|soon|still|such|ten|tenth|than|that|that|the|their|theirs|them|themselves|then|thence|there|therefore|these|they|third|thirteen|thirteenth|thirtieth|thirty|this|thither|those|though|thousand|thousandth|three|thrice|through|thus|till|to|today|tomorrow|too|towards|twelfth|twelve|twentieth|twenty|twice|two|under|underneath|unless|until|up|us|very|was|we|were|what|when|whence|where|whereas|which|while|whither|who|whom|whose|why|will|with|within|without|wo|would|yes|yesterday|yet|you|your|yours|yourself|yourselves|'re|'ve|n't|'ll|'twas|'em|y'|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|1|2|3|4|5|6|7|8|9|0)"
    files = glob.glob(dir_in+"*.txt")
    if len(files) > 0:
        list_of_dicts = list()
        for file in files:
            print(r"Tag counting file:", file)
            file_name = os.path.basename(file)
            text = open(file=file, encoding='utf-8', errors='ignore').read()
            text = re.sub(r"\n", r" ", text) #converts end of line in space
            words = re.split(r" +", text)
            #check if file is not empty
            if re.search(r"\w+", text):
                # ELF: Corrected an error in the MAT which did NOT ignore punctuation in token count (although comments said it did). Also decided to remove possessive s's, symbols, filled pauses and interjections (FPUH) from this count.
                # Shakir: list of words that match the given regex, then take its length as function words
                n_functionwords = len([word for word in words if re.search(r"\b" + function_words_re + r"_", word, re.IGNORECASE)])
                #print(n_functionwords)
                # EFL: Counting function words for lexical density
                # Shakir: list of words that do not containt SYM etc. + only if it is a word_TAG combination
                tokens = [word for word in words if not re.search(r"(_\s)|(\[\w+\])|(.+_\W+)|_-LRB-|_-RRB-|.+_SYM|_POS|_FPUH|_HYPH", word) if re.search(r"^\S+_\S+$", word)]
                # EFL: Counting total nouns for per 100 noun normalisation
                # Shakir: list of words that match the given regex, then take its length as total nouns 
                Ntotal = len([word for word in words if re.search(r"_NN\b", word)])
                # EFL: Approximate counting of total finite verbs for the per 100 finite verb normalisation
                # Shakir: list of words that match the given regex, then take its length as total verbs
                VBtotal = len([word for word in words if re.search(r"(_VPRT|_VBD|_VIMP|_MDCA|_MDCO|_MDMM|_MDNE|_MDWO|_MDWS)\b", word)])
                # ELF: I've decided to exclude all of these for the word length variable (i.e., possessive s's, symbols, punctuation, brackets, filled pauses and interjections (FPUH)):
                # Shakir: get the len of each word after splitting it from TAG, and making sure the regex punctuation does not match + only if it is a word_TAG combination
                list_of_wordlengths = [len(word.split('_')[0]) for word in words if not re.search(r"(_\s)|(\[\w+\])|(.+_\W+)|_-LRB-|_-RRB-|.+_SYM|_POS|_FPUH|_HYPH|_AFX|_NFP", word) if re.search(r"^\S+_\S+$", word)]
                # Shakir: total length of characters / length of the list which represents the length of each word, i.e. tokens just as above
                average_wl = sum(list_of_wordlengths) / len(list_of_wordlengths) # average word length
                lex_density = (len(tokens) - n_functionwords) / len(tokens) # ELF: lexical density
                #print(len(tokens), lex_density)
                ttr = get_ttr(tokens, n_tokens) # Shakir calculate type token ratio
                # Shakir: get tags only, remove words and exclude certain (non-meaningful) tags from count
                # ELF: The list of tags for which no counts will be returned can be found here.
                # The following tags are excluded by default because they are "bin" tags designed to remove problematic tokens from other categories: LIKE and SO
                # Note: if interested in counts of punctuation marks, "|_\W+" should be deleted in this line.
                # Note: _WQ are removed because they are duplicates of WHQU (WHQU are tagged onto the WH-words whereas QUWU onto the question marks themselves).
                tags = [re.sub(r"^.*_", "", word) for word in words if not re.search(r"_LS|_\W+|_WP\\b|_FW|_SYM|_MD\\b|_VB\\b|_WQ|_LIKE|_SO", word)]
                # Shakir: get a dictionary of tags and frequency of each tag using collections.Counter on tags list
                tag_freq = dict(collections.Counter(tags))
                # Shakir: sort tag_freq
                tag_freq = dict(sorted(tag_freq.items()))
                temp_dict = {'Filename': file_name, 'Words': len(tokens), 'AWL': average_wl, 'TTR': ttr, 'LDE': lex_density, 'Ntotal': Ntotal, 'VBtotal': VBtotal}
                # update temp dict with tag freq
                temp_dict.update(tag_freq)
                list_of_dicts.append(temp_dict)
        print("writing statistics...")
        df = pd.DataFrame(list_of_dicts).fillna(0)
        features_to_be_removed_from_final_table_existing = [f for f in features_to_be_removed_from_final_table if f in df.columns]
        df = df.drop(columns=features_to_be_removed_from_final_table_existing) #drop unnecessary features
        df = sort_df_columns(df).sort_values(by=['Filename']) #sort df columns
        df.round().drop(columns=['Ntotal', 'VBtotal']).to_csv(dir_out+"counts_raw.csv", index=False)
        #df = pd.read_excel(dir_out+"counts_raw.csv")
        get_complex_normed_counts(df).drop(columns=['Ntotal', 'VBtotal']).round(4).to_csv(dir_out+"counts_mixed_normed.csv", index=False)
        get_wordbased_normed_counts(df).drop(columns=['Ntotal', 'VBtotal']).round(4).to_csv(dir_out+"counts_word-based_normed.csv", index=False)
        print("finished!")
    else:
        print("It appears there are no files to count tags from. Maybe you did not input the correct path?")
    # with open(file=dir_out+"tokens.txt", mode='w', encoding='utf-8') as f:
    #     f.write("\n".join(tags))
    #     break    

if __name__ == "__main__":
    input_dir = r"D:\Corpus Related\Corpora\Pakistani English Historical\corpus\\"
    # download Stanford CoreNLP and unzip in this directory. See this page #https://stanfordnlp.github.io/stanza/client_setup.html#manual-installation
    # direct download page https://stanfordnlp.github.io/CoreNLP/download.html
    output_main = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_tagged/"
    output_stanford = output_main + "StanfordPOS_Tagged/"
    output_MD = output_main + "MFTE_Tagged/"
    output_stats = output_main + "Statistics/"
    ttr = 400
    # tag_stanford(nlp_dir, input_dir, output_stanford)
    t_0 = timeit.default_timer()
    #tag_stanford_stanza(input_dir, output_stanford)
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print("Time spent on tagging process (micro seconds):", elapsed_time)
    #tag_MD(output_stanford, output_MD, extended=True)
    tag_MD_parallel(output_stanford, output_MD, extended=True)
    do_counts(output_MD, output_stats, ttr)

# if __name__ == "__main__":
#     input_dir = r"/Users/Elen/Documents/PhD/Publications/2023_Shakir_LeFoll/MFTE_python/MFTE_Eval/COCA/COCA_test2/"
#     # download Stanford CoreNLP and unzip in this directory. See this page #https://stanfordnlp.github.io/stanza/client_setup.html#manual-installation
#     # direct download page https://stanfordnlp.github.io/CoreNLP/download.html
#     output_main = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_tagged/"
#     output_stanford = output_main + "StanfordPOS_Tagged/"
#     output_MD = output_main + "MFTE_Tagged/"
#     output_stats = output_main + "Statistics/"
#     ttr = 400
#     # record start time
#     t_0 = timeit.default_timer()
#     #tag_stanford_stanza(input_dir, output_stanford)
#     #tag_stanford(nlp_dir, input_dir, output_stanford)
#     t_1 = timeit.default_timer()
#     elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
#     print("Time spent on tagging process (micro seconds):", elapsed_time)
#     #tag_MD(output_stanford, output_MD, extended=True)
#     #tag_MD_parallel(output_stanford, output_MD, extended=True)
#     do_counts(output_MD, output_stats, ttr)
