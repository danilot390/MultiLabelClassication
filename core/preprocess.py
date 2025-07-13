import pandas as pd 
import re
import yaml

from sklearn.preprocessing import LabelEncoder

from core.utils import Config


def get_input_data()->pd.DataFrame:
    df1 = pd.read_csv(Config['APP_GALLERY'], skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df2 = pd.read_csv(Config['PURCHASING'], skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df = pd.concat([df1, df2])
    df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].values.astype('U')
    df[Config['TICKET_SUMMARY']] = df[Config['TICKET_SUMMARY']].values.astype('U')
    df["y"] = df[Config['CLASS_COL']]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df

def de_duplication(data: pd.DataFrame):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            [r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            [r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = r"(On.{30,60}wrote:)"
    pattern_3 = r"(Re\s?:|RE\s?:)"
    pattern_4 = r"(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = r"(\s?\*\*\*\*\*\(PHONE\))*$"

    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # -------- start processing ticket data

    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        #print(t)
        df = data.loc[data['Ticket id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df[Config['INTERACTION_CONTENT']]:

            # print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            #print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated
    data[Config['INTERACTION_CONTENT']] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data

def noise_remover(df: pd.DataFrame):
    noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config['TICKET_SUMMARY']] = df[Config['TICKET_SUMMARY']].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].str.lower()
    noise_1 = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"dear ((customer)|(user))",
        r"dear",
        r"(hello)|(hallo)|(hi )|(hi there)",
        r"good morning",
        r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        r"thank you for contacting us",
        r"thank you for your availability",
        r"thank you for providing us this information",
        r"thank you for contacting",
        r"thank you for reaching us (back)?",
        r"thank you for patience",
        r"thank you for (your)? reply",
        r"thank you for (your)? response",
        r"thank you for (your)? cooperation",
        r"thank you for providing us with more information",
        r"thank you very kindly",
        r"thank you( very much)?",
        r"i would like to follow up on the case you raised on the date",
        r"i will do my very best to assist you"
        r"in order to give you the best solution",
        r"could you please clarify your request with following information:"
        r"in this matter",
        r"we hope you(( are)|('re)) doing ((fine)|(well))",
        r"i would like to follow up on the case you raised on",
        r"we apologize for the inconvenience",
        r"sent from my huawei (cell )?phone",
        r"original message",
        r"customer support team",
        r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        r"canada, australia, new zealand and other countries",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)"]
    for noise in noise_1:
        #print(noise)
        df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].replace(noise, " ", regex=True)
    df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].replace(r'\s+', ' ', regex=True).str.strip()
    #print(df.y1.value_counts())
    good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    df = df.loc[df.y1.isin(good_y1)]
    #print(df.shape)
    return df

def label_encoder(df):
    # Encode categorical target variables into numeric values
    label_encoders = {}
    for col in [Config['INTENT_COL'], Config['TONE_COL'], Config['RESOLUTION_COL']]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return label_encoders

def translate_to_en(texts:list[str]):
    from langdetect import detect_langs
    from transformers import pipeline
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    model_id = "facebook/m2m100_418M"  # M2M-100 model for multilingual translation
    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
    model = M2M100ForConditionalGeneration.from_pretrained(model_id)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, tgt_lang="en"
                          , src_lang="auto")
    result = []

    for text in texts:
        if text == "":
            result.append(text)
            continue

        # Detect language
        lang = detect_langs(text)[0].lang
        if lang == "en":
            result.append(text)
        else:
            # Translate to English
            translation = translator(text, src_lang=lang, tgt_lang="en")
            result.append(translation[0]['translation_text'])

    return result