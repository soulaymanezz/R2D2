import speech_recognition as sr
import pyttsx3
import nltk
import warnings
import datetime
import webbrowser
import time

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
r = sr.Recognizer()

# Connexion Processing
'''HOST = '127.0.0.1'  # Symbolic name meaning all available interfaces
PORT = 61915  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
conn.close()'''

#French version
def francais():
    engine = pyttsx3.init()
    fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
    # Use female french voice
    engine.setProperty('voice', fr_voice_id)
    def there_exists(terms):
        for term in terms:
            if term in voice_data:
                return True

    def speak(text):
        engine.say(text)
        engine.runAndWait()
    def record_audio(ask=""):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            if ask:
                speak(ask)
            audio = r.listen(source)
            try:
                voice_data = r.recognize_google(audio, language='fr')
                print(voice_data)

            except:
                voice_data = record_audio("Pardon, pouvez-vous répéter, je vous écoute")
                return voice_data
            return voice_data

    def respond(voice_data):
        # quit
        if there_exists(["au revoir", "ok bye", "stop", "bye"]):
            speak("Votre robot d'assistance s'arrête, au revoir")
            exit()

        # Elearning
        if there_exists(['elearning', 'ouvrir elearning']):
            webbrowser.open_new_tab("https://elearning.emines.um6p.ma/")
            speak("Vous avez l'accés à Elearning")
            time.sleep(3)

        # Oasis
        if there_exists(['oasis', 'ouvrir oasis']):
            a = "https://emines.oasis.aouka.org/"
            speak("Vous avez l'accés à Oasis")
            time.sleep(3)

        # Outlook
        if there_exists(['Outlook','outlook', 'ouvrir Outlook']):
            webbrowser.open_new_tab("https://www.office.com/")
            speak("Vous avez l'accés à Outlook")
            time.sleep(3)

        # Emines website
        if there_exists(['site emines', 'émine']):
            webbrowser.open_new_tab("https://www.emines-ingenieur.org/")
            speak("Vous avez l'accés au site d'EMINES")
            time.sleep(3)

        # visite UM6P
        if there_exists(["je veux visiter virtuellement l'université", "visite virtuelle"]):
            webbrowser.open_new_tab("https://alpharepgroup.com/um6p_univer/models.php")
            speak("Bienvenue à la visite virtuelle de um6p ")
            time.sleep(3)

        # time
        if there_exists(['quelle heure est-il', 'heure']):
            strTime = datetime.datetime.now().strftime("%H""heures""%M")
            speak(f"Il est {strTime}")

        # presentation
        if there_exists(['présente-toi', 'qui est tu']):
            speak("Je suis votre robot d'assistance" " Réalisé par les étudiants du quatrième année d'EMINE"
                  "Je suis un projet mécatronique pour rendre service aux visiteurs de l'université mohamed 6 polytechnique")
            time.sleep(3)

        # google
        if there_exists(['google', 'ouvrir google']):
            webbrowser.open_new_tab("https://www.google.com")
            speak("Vous avez l'accés à Google")
            time.sleep(3)

        # localisation google maps
        if there_exists(["où suis-je exactement", "où suis-je", "où je suis ", "où je suis exactement", "localisation"]):
            webbrowser.open_new_tab("https://www.google.com/maps/search/Where+am+I+?/")
            speak("Selon Google maps, vous devez être quelque part près d'ici")
            time.sleep(3)

        # météo
        if there_exists(["météo", "combien fait-il de degrés maintenant", "degré"]):
            search_term = voice_data.split("for")[-1]
            webbrowser.open_new_tab(
                "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5")
            speak("Voici ce que j'ai trouvé pour la météo sur Google")
            time.sleep(3)

        # BDD
        Noms =['Nicolas', 'Cheimanoff', 'Khadija', 'AitHadouch', 'Frédéric',
            'Fontane', 'Bouchikhi', 'Reda', 'Fatiha','Elabdellaoui']
        jobs = ['directeur','logistique','responsable','recherche']
        l1 = ['Cheimanoff', 'Nicolas','Directeur','Nicolas.CHEIMANOFF@emines.um6p.ma','B']
        l2 = ['Fontane', 'Frederic',"Directeur de l'enseignement",'Frederic.FONTANE@emines.um6p.ma','C']
        l3 = ['Elabdellaoui', 'Fatiha','Responsable de scolarite','Fatiha.ELABDELLAOUI@emines.um6p.ma','C']
        l4 = ['AitHadouch', 'Khadija','Assistante de direction','Khadija.AITHADOUCH@emines.um6p.ma','B']
        l5 = ['Bouchikhi', 'Reda','Responsable logistique','Reda.Bouchikhi@emines.um6p.ma','D']
        if there_exists(Noms):
            list = voice_data.split()
            term = set(list) & set(Noms)
            term0 = " ".join(term)
            if term0 in Noms:
                answer = record_audio("voulez vous savoir qui est " + term0)
                respond(answer)
                if 'oui' in answer:
                    if term0 in ['Nicolas','Cheimanoff']:
                        l = l1
                    elif term0 in ['Frédéric', 'Fontane']:
                        l = l2
                    elif term0 in ['Fatiha', 'Elabdellaoui']:
                        l = l3
                    elif term0 in ['Khadija', 'AitHadouch']:
                        l = l4
                    elif term0 in ['Reda', 'Bouchikhi']:
                        l = l5
                    engine = pyttsx3.init()
                    engine.say(l[1] + l[0])
                    engine.say(l[2] + "de l'EMINES  ")
                    engine.say(" son bureau se  trouve dans le bloc   " + l[4])
                    engine.say(" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                    engine.runAndWait()
        elif there_exists(jobs):
            list1 = voice_data.split()
            term1 = set(list1) & set(jobs)
            job = " ".join(term1)
            list2 = job.split()
            list2.reverse()
            print(list2)
            if any(item in jobs for item in list2):
                answer = record_audio("voulez vous savoir qui est " + job + "de l'EMINES")
                respond(answer)
                if 'oui' in answer:
                    if any(item in ['directeur',"l'enseignement"] for item in list2):
                        l = l1
                    if any(item in ['logistique','responsable'] for item in list2):
                        l = l5
                engine = pyttsx3.init()
                engine.say(l[1] + l[0])
                engine.say(l[2] + "de l'EMINES  ")
                engine.say(" son bureau se  trouve dans le bloc   " + l[4])
                engine.say(" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                engine.runAndWait()
        #else:
            #voice = record_audio("Je vous ai pas compris, pouvez vous repeter?")
            #respond(voice)

    speak("Bienvenus à l'université Mohammed 6 polytechnique, je suis votre robot d'assistance. Quelle est votre question ?")
    while True:
        voice_data = record_audio("je vous écoute")
        respond(voice_data)
        time.sleep(10)
        if there_exists(["j'ai besoin de votre aide"]):
            voice_data = record_audio("je vous écoute")
            print(voice_data, ...)
            respond(voice_data)
        else:
            time.sleep(5)
            voice_data = record_audio("Avez-vous besoin de mon aide")
            #voice_data = record_audio("je vous écoute")
            print(voice_data, ...)
            if there_exists(['oui']):
                voice_data = record_audio("je vous écoute")
                print(voice_data, ...)
                respond(voice_data)
                continue
            if there_exists(['non','merci']):
                speak("Votre robot assistant s'arrête, au revoir")
                print("Votre robot assistant s'arrête, au revoir")
                break
francais()