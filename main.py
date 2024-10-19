import sys 
sys.path.append('checkbox_detector/')
import os
import omicacr
import numpy as np
import matplotlib.pyplot as plt
import postprocessing
import json
import requests
import openai
import time

path_escaneados = os.path.abspath("../limpio/")
examples_escaneados = os.listdir(path_escaneados)

def test_omicacr(to_be_tested=None):
    '''
    Test the omicacr module 

    Parameters
    ----------
    to_be_tested : list of int, optional
        List of indexes of the examples to be tested. If None, all the examples will be tested.

    Returns
    -------
    None
    '''
    if to_be_tested is None:
        to_be_tested = [
            i for i, e in enumerate(examples_escaneados) if e.endswith(".pdf")
        ]

    # Dictionary with the results
    results = {}

    print([name for i, name in enumerate(examples_escaneados) if i in to_be_tested])

    # read json to dict
    format_json = {}
    with open('json_base.json', 'r', encoding='utf-8') as f:
        format_json = json.load(f)
    
    format_json = json.dumps(format_json, indent=4, ensure_ascii=False)

    for index in to_be_tested:
        example_path = os.path.join(path_escaneados, examples_escaneados[index])
        imgs = omicacr.extract_img_from_pdf(example_path)
        # show images
        correction = {}
        for n_page, img in enumerate(imgs):
            # img = omicacr.preprocess_input(img)
            sections = omicacr.get_document_sections(img, show=False)
            parsed_document = postprocessing.process_and_parse_data(img, sections)
            json_data_ = json.dumps(parsed_document, indent=4, ensure_ascii=False)
            
            # print(json_data_)
            # wait some seconds to avoid openai error
            time.sleep(5)

            openai.api_key = "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"
            user_input = "Corrige el siguiente JSON obtenido de un documento historial clinico: \n ```json\n" + json_data_ + "```."
            messages = [
                {"role": "system", "content": "Eres un asistente que corrige faltas de ortografia en español, das formato y corriges frases que no tengan sentido o tengan errores gramaticales. Contesta solo con el contenido y formato del JSON"},
                {"role": "user", "content": user_input}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
            )
            
            try:
                gpt_respuesta = response["choices"][0]["message"]['content'].strip().split("```json")[1].split("```")[0]
                print(gpt_respuesta)
                correction.update(json.loads(gpt_respuesta))            
            except:
                print("Error en la pagina: ",n_page+1," de ", len(imgs)," de ",examples_escaneados[index])
                print(response)

            results.update(parsed_document)

            print("Listo pagina: ",n_page+1," de ", len(imgs)," de ",examples_escaneados[index])
        
        # to json
        json_data = json.dumps(results, indent=4, ensure_ascii=False)
        # print(json_data)

        if not os.path.exists("data_extracted"):
            os.makedirs('data_extracted')
        
        # save json
        with open('data_extracted/data_extracted_' + examples_escaneados[index] + '.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        with open('data_extracted/data_extracted_corrected' + examples_escaneados[index] + '.json', 'w', encoding='utf-8') as f:
            json.dump(correction, f, ensure_ascii=False, indent=4)
       
    # omicacr.close_session()

if __name__ == "__main__":
    print("Prueba OCR")
    test_omicacr()
