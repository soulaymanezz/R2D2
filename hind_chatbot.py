"""import pyttsx3
# Initialize the converter
converter = pyttsx3.init()
voices = converter.getProperty('voices')
for voice in voices:
    # to get the info. about various voices in our PC
    print("Voice:")
    print("ID: %s" % voice.id)
    print("Name: %s" % voice.name)
    print("Age: %s" % voice.age)
    print("Gender: %s" % voice.gender)
    print("Languages Known: %s" % voice.languages)"""

#Import the libraries
import nltk
import numpy as np
import wave
import math
import contextlib
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import speech_recognition as sr
import os
from os import path
import pyttsx3
#download the punkt package
nltk.download('punkt', quiet=True)
#global variables
r = sr.Recognizer()
list_max = []
Noms = []
s=0 #the number that helps to know when excuting the voice function what i should excute the main boucle or a boucle inside it
def geo():
    print('got u')
def boucle():
    #text1 is the one we get from the voice function
    words1 = text1.split() #the sentence to a list of words
    converted_words1 = [x.upper() for x in words1] #the words' list en majuscule
    words = converted_words1 #new list of words with all the words majuscule
    print(words)
    exit_list = ['REVOIR', 'BYE', 'PROCHAINE']
    text2 = set(words) & set(exit_list) #setting the intersection between the two lists words and exit_list
    str_val = " ".join(text2) #making the intersection words a string to use it
    text02 = str_val #the new text variable we going to work with
    #two conditions whether the visitor wanna leave or complete
    if text02 in exit_list:
        sum = "0"
        text_say = "au revoir, et bienvenue à emines"
        engine = pyttsx3.init()
        engine.say(text_say)
        engine.runAndWait()
        sum = "1"
    else:
        #define the list we using to match the departement
        list1 = ['RÉCEPTION', 'ENTRÉE', 'BLOC', 'A']
        list2 = ['DÉPARTEMENT', 'BLOC', 'NICOLAS', 'CHEIMANOFF', 'KHADIJA', 'AITHADOUCH', 'FRÉDÉRIC', 'DIRECTION',
                 'DIRECTION EMINES', 'NICO', 'SAAD', 'KHATAB', 'B']
        list3 = ['DÉPARTEMENT', 'FATIHA', 'ABDELAOUI', 'C', 'BLOC', 'FOYER', 'SCOLARITÉ','ZINEB']
        list4 = ['DÉPARTEMENT', 'BLOC', 'D', 'REDA', 'BOUCHIKHI', 'HAJAR', 'KHOUKH']
        list5 = ['POLE', ' SANTÉ', 'BLOC', 'E', 'MÉDECIN', 'SANITAIRE', 'INFIRMERIE']
        list6 = ['ETECH', 'LABO', 'LABORATOIRE', 'BLOC']
        a = []
        hind = ['Réception', 'département direction', 'département de scolarité',
                'département logistique', 'département santé', 'E-tech'] #list of the departement names
        #setting the matching words between the original words list we got from the visitor text said and the lists of departements
        list7 = sorted(set(list1) & set(words), key=lambda k: list1.index(k))
        a.append(list7)
        list8 = sorted(set(list2) & set(words), key=lambda k: list2.index(k))
        a.append(list8)
        list9 = sorted(set(list3) & set(words), key=lambda k: list3.index(k))
        a.append(list9)
        list10 = sorted(set(list4) & set(words), key=lambda k: list4.index(k))
        a.append(list10)
        list11 = sorted(set(list5) & set(words), key=lambda k: list5.index(k))
        a.append(list11)
        list12 = sorted(set(list6) & set(words), key=lambda k: list6.index(k))
        a.append(list12)
        # print(a)
        max(a) #the list with a max of matching words
        print(max(a))
        for i in range(6):
            if max(a) == a[i]: #for i bind with max list matched get it and then from the department list names take the name corresponding to it
                print(hind[i])
                global département_déplacement
                département_déplacement = hind[i]
                Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
                        'HAJAR', 'KHATAB',
                        'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
                        'NICO', 'SAAD','ORCHI'] #list of all the  emines' staff names
                Noms1 = ['FRÉDÉRIC','NICOLAS', 'KHADIJA', 'KHATAB', 'SAAD', 'CHEIMANOFF', 'AITHADOUCH', 'NICO']  # direction
                Noms2 = ['ZINEB', 'KHOUKH', 'ABDELAOUI', 'FATIHA', 'KAWTAR']  # scolarité
                Noms3 = ['REDA', 'HAJAR', 'BOUCHIKHI', 'ORCHI']  # logistique
                Noms4 = ['INFIRMIÈRE', 'MÉDECIN']  # health center
                global NOMS
                if département_déplacement == 'département direction':
                    NOMS = Noms1
                elif département_déplacement == 'département de scolarité':
                    NOMS = Noms2
                elif département_déplacement == 'département logistique':
                    NOMS = Noms3
                elif département_déplacement == 'département santé':
                    NOMS = Noms4
                term0 = set(words) & set(Noms)
                str_val = " ".join(term0)
                global term1
                term1 = str_val
                print(term1)
                if term1 in words:  # see if one of the words in the sentence is the word we want
                    sum = "0"
                    text_term1_say = "voulez vous allez au " + département_déplacement + " chez " + term1 #term1 is the name of the person we want
                    engine = pyttsx3.init()
                    engine.say(text_term1_say)
                    engine.runAndWait()
                    sum = "1"
                    global s
                    s = 2
                    print(s)
                    register()
                else:
                    sum = "0"
                    text_term_say = "voulez vous allez au" + département_déplacement + "chez une personne précise?"
                    engine = pyttsx3.init()
                    engine.say(text_term_say)
                    engine.runAndWait()
                    sum = "1"
                    s = 3
                    register()
def boucle1():
    words2 = text22.split()
    print(words2)
    words22 = [x.upper() for x in words2]  # new list of words with all the words majuscule
    print(words22)
    rsp = " ".join(words22)  # making the intersection words a string to use it
    if 'OUI' in rsp :
        print('ok')
        print(NOMS)
        if term1 in NOMS:
            sum = "0"
            text__say = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
            engine = pyttsx3.init()
            engine.say(text__say)
            engine.runAndWait()
            sum = "1"
            #geo(term1)
        else:
            sum = "0"
            text__say_else = "le nom et le département que vous voulez ne sont pas dépendants "
            engine = pyttsx3.init()
            engine.say(text__say_else)
            engine.runAndWait()
            sum = "1"
            main()
    else:
        sum = "0"
        text__say1 = "je vous est pas bien compris, pouvez redire "
        engine = pyttsx3.init()
        engine.say(text__say1)
        engine.runAndWait()
        sum = "1"
        global s
        s = 2
        register()
def boucle2():
    Text = text3.split()
    print(Text)
    words3 = [x.upper() for x in Text]  # new list of words with all the words majuscule
    print(words3)
    text33 = " ".join(words3)
    if 'OUI' in text33:
        sum = "0"
        text0 = "chez qui?"
        engine = pyttsx3.init()
        engine.say(text0)
        engine.runAndWait()
        sum = "1"
        global s
        s=4
        register()
    else:
        sum = "0"
        engine = pyttsx3.init()
        engine.say("D'accord, suivez moi je vais vous emmenez au " + département_déplacement)
        engine.runAndWait()
        sum = "1"
        # do_stuff()
def boucle3():
    words4 = text4.split()
    print(words4)
    z = [x.upper() for x in words4]
    Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
            'HAJAR', 'KHATAB',
            'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
            'NICO', 'SAAD']  # list of all the  emines' staff names
    term4 = set(z) & set(Noms)
    str_val = " ".join(term4)
    global term44
    term44 = str_val
    print(term44)
    if term44 in Noms:
        term41 = term44.lower()
        sum = "0"
        text_say3 = "voulez vous allez au" + département_déplacement + "chez " + term41
        engine = pyttsx3.init()
        engine.say(text_say3)
        engine.runAndWait()
        sum = "1"
        global s
        s=5
        register()
    else:
        sum = "0"
        engine = pyttsx3.init()
        engine.say("je trouve pas la personne que vous voulez")
        engine.runAndWait()
        sum = "1"
        main()
def boucle4():
    word5 = text5.split()
    print(word5)
    word5 = [x.upper() for x in word5]  # new list of words with all the words majuscule
    print(word5)
    text55 = " ".join(word5)
    if 'OUI' in text55:
        print(NOMS)
        if term44 in NOMS:
            sum = "0"
            text_say_term44 = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
            engine = pyttsx3.init()
            engine.say(text_say_term44)
            engine.runAndWait()
            sum = "1"
            # do_stuff()
        else:
            sum = "0"
            text__say_else0 = "le nom et le département que vous voulez ne sont pas dans le même emplacement "
            engine = pyttsx3.init()
            engine.say(text__say_else0)
            engine.runAndWait()
            sum = "1"
            main()
    else:
        sum = "0"
        text0 = "je trouve pas la personne que vous voulez"
        engine = pyttsx3.init()
        engine.say(text0)
        engine.runAndWait()
        sum = "1"
        main()
"""the function that changes the audio file contain to a text and calls the function
with all the possibilities to answer called boucle"""
#def function voice; audio to text
def voice():
    print('voicetotext')
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "C:\\filtered1.wav")
    # use the audio file as the audio source
   
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
    # recognize speech using Google Speech Recognition
        TEXT = r.recognize_google(audio, language="fr-FR")
        print(TEXT)
        try:
            global s
            if s==0:
                text0 = TEXT
                print(text0)
                """taking the response from the visitor, the robot even start explaining and then runs the main function or just run it is
                the visitor wanna pass the explanation"""
                Text0 = text0.split()
                list_continuer = ['continue','continuer','terminer','entendre']
                text001 = set(Text0) & set(list_continuer)  # setting the intersection between the two lists words and exit_list
                str_val = " ".join(text001)  # making the intersection words a string to use it
                text001 = str_val  # the new text variable we going to work with
                # two conditions whether the visitor wanna leave or complete
                try:
                    if text001 in list_continuer:
                        sum = "0"
                        text01 = "d'accord, je commencerais :Au niveau du rez de chaussée se trouve cinq département; premièrement la réception qui se trouve au bloc A: " \
                                "si vous voulez mieux connaitre université deuxièment la direction qui se trouve au bloc B: se trouve le bureau " \
                                "du directeur Nicolas Cheimanoff , son assistante Khadiija Aitahadouch et le bureau de Saad Aitkhatab le " \
                                "responsable de communication de EMINES troixièment la scolarité qui se trouve au bloc C: elle se trouve en face" \
                                " du foyer des élèves, se trouve le bureau de Fatiha Alabdelaoui responsable de scolarité de EMINES, Zineb " \
                                "Elkhoukh assistante du directeur d'éducation et de la recherche, et messieur Orchi responsable d'impression " \
                                "quatrièment la logistique qui se trouve au bloc D: il se trouve le bureau de Reda Elbouchikhi responsable " \
                                "hébergement son assistante hajar Azerkouk cinquièment le pôle santé: ou se trouve le médecin et les infirmières " \
                                "et dernièrement E-tech: le club de robotique Emines si vous voulez consulter les projets effectués par nos " \
                                "étudiants."
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 178)
                        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
                        # Use female french voice
                        engine.setProperty('voice', fr_voice_id)
                        engine.say(text01)
                        engine.runAndWait()
                        sum = "1"
                        main()
                    else:
                        main()
                # os.remove("filtered1.wav")
                    print("File Removed!")
                except sr.RequestError:
                    sum = "0"
                    text_say__0_try = "erreur"
                    engine = pyttsx3.init()
                    engine.say(text_say__0_try)
                    engine.runAndWait()
                    sum = "1"
                    register()
                except sr.UnknownValueError:
                    sum = "0"
                    text_say__0_try = "je vous ai pas bien entendu, pouvez vous répètez, merci"
                    engine = pyttsx3.init()
                    engine.say(text_say__0_try)
                    engine.runAndWait()
                    sum = "1"
                    register()
            elif s == 1:
                global text1
                text1 = TEXT
                print(text1)
                boucle()
                os.remove("filtered1.wav")
                print("File Removed!")
            elif s == 2:
                global text22
                text22 = TEXT
                print(text22)
                boucle1()
                os.remove("filtered1.wav")
                print("File Removed!")
            elif s==3:
                global text3
                text3 = TEXT
                print(text3)
                boucle2()
                os.remove("filtered1.wav")
                print("File Removed!")
            elif s==4:
                global text4
                text4 = TEXT
                print(text4)
                boucle3()
                os.remove("filtered1.wav")
                print("File Removed!")
            elif s==5:
                global text5
                text5 = TEXT
                print(text5)
                boucle4()
                os.remove("filtered1.wav")
                print("File Removed!")
        except sr.UnknownValueError:
            sum = "0"
            text_say__0 = "je vous ai pas bien entendu, pouvez vous répètez, merci"
            engine = pyttsx3.init()
            engine.say(text_say__0)
            engine.runAndWait()
            sum = "1"
            register()
        except sr.RequestError:
            sum = "0"
            text_say__0_try = "erreur"
            engine = pyttsx3.init()
            engine.say(text_say__0_try)
            engine.runAndWait()
            sum = "1"
            register()
"""the function that takes the voices from the microphone register it in a wav file and then filter it and call the function
that changes the audio file contain into a text called voice"""
#def register; register voice and filter it
def register():
    print('register')
    freq = 41000
    #i should calculate the duration using the volume
    duration = 6
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    write("C:\\test1.wav", freq, recording)
    fname = 'C:\\test1.wav'
    outname = 'C:\\filtered1.wav'
    cutOffFrequency = 200.0
    def running_mean(x, windowSize):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
    def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
        if sample_width == 1:
            dtype = np.uint8  # unsigned char
        elif sample_width == 2:
            dtype = np.int16  # signed 2-byte short
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")
        channels = np.frombuffer(raw_bytes, dtype=dtype)
        if interleaved:
            # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
            channels.shape = (n_frames, n_channels)
            channels = channels.T
        else:
            # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
            channels.shape = (n_channels, n_frames)
        return channels
    data, samplerate = sf.read(fname)
    sf.write(fname, data, samplerate, subtype='PCM_16')
    with contextlib.closing(wave.open(fname, 'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()
        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames * nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
        # get window size
        freqRatio = (cutOffFrequency / sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)
        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)
        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()
        voice()
"""the main function here is the first one calles in the programm and it counts n the number of no responses once n is higher or
equal to 2 due to the bad quality of voice...etc, the visitor gonna be directed to tape his direction on an interface """
#the main function; count n and execute the register function for n<=1 / n>=2 --> interface
n = 0; #nombre de reponse = non càd le nombre ou le main fonction était répétée
def main():
    global n
    print(n)
    n += 1
    print(n)
    if n <= 2:# n is the nombre of tries and said no as response
        text_main0 = "qu'elle est votre question?"
        engine = pyttsx3.init()
        engine.say(text_main0)
        engine.runAndWait()
        global s
        s=1
        print(s)
        register()
    else:
        text_main1 = "je préfére que vous tapez votre question, afin que je puisse vous comprendre"
        engine = pyttsx3.init()
        engine.say(text_main1)
        engine.runAndWait()
        #ajouter une interface
def start():
    # obtain audio from the microphone
    """the starting text where the robot represente himself and offer to explain more about the school"""
    text00 = "Bonjour chers visiteurs je suis votre robot d'assistance, je suis destiné à vous aider à se déplacer au sein de " \
            "l'EMINES et vous diriger vers votre destination et aussi répondre à vos questions."
    text000="je commencera par vous décrire le plan de l'école pour que je puisse vous aider efficacement à se déplacer,      mais  vous pouvez dépasser " \
            "cette partie descriptive durant cette étape en disons je passe,   sinon et si vous voulez l'entendre disez simplement je continue, "
    text0000="je vous renseigne aussi qu'une fois mon micro est ouvert pour vous entendre mon interface devienne verte."
    engine = pyttsx3.init()
    engine.setProperty("rate",500)
    fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
    # Use female french voice
    engine.setProperty('voice', fr_voice_id)
    engine.say(text00)
    engine.say(text000)
    engine.say(text0000)
    engine.runAndWait()
    register()
start()