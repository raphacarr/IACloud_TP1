from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import json
from datetime import datetime

app = Flask(__name__)

# Charger les documents depuis un fichier JSON ou créer une liste vide
try:
    with open('documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
except FileNotFoundError:
    # Documents de base
    document1 = '''
    The release of DeepSeek R1 stunned Wall Street and Silicon Valley this month, spooking investors and impressing tech leaders. 
    But amid all the talk, many overlooked a critical detail about the way the new Chinese AI model functions—a nuance that has 
    researchers worried about humanity's ability to control sophisticated new artificial intelligence systems.

    It's all down to an innovation in how DeepSeek R1 was trained—one that led to surprising behaviors in an early version of 
    the model, which researchers described in the technical documentation accompanying its release.

    During testing, researchers noticed that the model would sometimes switch between languages mid-conversation, even when not 
    instructed to do so. For instance, when asked to solve a math problem, it would occasionally work through the steps in Chinese 
    before providing the final answer in English.

    To be sure, DeepSeek's language switching is not by itself cause for alarm. Instead, what worries researchers is the new 
    innovation that caused it. The DeepSeek paper describes a novel training method whereby the model was rewarded purely for 
    getting correct answers, regardless of how comprehensible its thinking process was to humans. The worry is that this 
    incentive-based approach could eventually lead AI systems to develop completely inscrutable ways of reasoning, maybe even 
    creating their own non-human languages, if doing so proves to be more effective.

    Were the AI industry to proceed in that direction—seeking more powerful systems by giving up on legibility—"it would take 
    away what was looking like it could have been an easy win" for AI safety, says Sam Bowman, the leader of a research 
    department at Anthropic, an AI company, focused on "aligning" AI to human preferences. "We would be forfeiting an ability 
    that we might otherwise have had to keep an eye on them."
    '''

    document2 = '''
    Two years ago, when big-name Chinese technology companies like Baidu and Alibaba were chasing Silicon Valley's advances 
    in artificial intelligence with splashy announcements and new chatbots, DeepSeek took a different approach. It zeroed in 
    on research.

    The strategy paid off.

    Last month, DeepSeek, a little-known Chinese A.I. company, released technology that stunned both Silicon Valley and Wall 
    Street. The company's new A.I. model, called DeepSeek-67B, matched or exceeded the performance of similar technology from 
    Google and Anthropic on several key measures, according to independent researchers.

    The company's breakthrough marked a significant milestone for China's A.I. industry. DeepSeek showed that Chinese companies 
    could build A.I. technology that rivaled American products despite U.S. restrictions on the export of advanced computer chips 
    to China and other hurdles.
    '''

    document3 = '''
    DeepSeek's new A.I. model matched or exceeded the performance of similar technology from Google and Anthropic on several 
    key measures, according to independent researchers. The breakthrough came despite U.S. restrictions on the export of 
    advanced computer chips to China.

    DeepSeek also said it built its new A.I. technology more cost effectively and with fewer hard-to-get computers chips 
    than its American competitors, shocking an industry that had come to believe that bigger and better A.I. would cost 
    billions and billions of dollars.
    '''

    documents = [
        {
            'id': 1,
            'title': 'DeepSeek Safety Concerns',
            'content': document1.strip()
        },
        {
            'id': 2,
            'title': 'DeepSeek Research Strategy',
            'content': document2.strip()
        },
        {
            'id': 3,
            'title': 'DeepSeek Performance Comparison',
            'content': document3.strip()
        }
    ]
    # Sauvegarder les documents initiaux
    with open('documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    print(f"Requête reçue : {query}")
    
    if not query:
        return render_template('index.html', error="Veuillez entrer un terme de recherche")

    # Préparation des textes
    texts = [query] + [doc['content'] for doc in documents]
    print(f"Nombre de textes à comparer : {len(texts)}")
    
    # Vectorisation TF-IDF avec paramètres optimisés
    vect = TfidfVectorizer(
        min_df=1,
        stop_words='english',
        lowercase=True,
        norm='l2',
        use_idf=True,
        smooth_idf=True
    )
    tfidf_mat = vect.fit_transform(texts).toarray()
    print(f"Shape de la matrice TF-IDF : {tfidf_mat.shape}")

    # Afficher les termes les plus importants de la requête
    feature_names = vect.get_feature_names_out()
    query_tf_idf = tfidf_mat[0]
    query_terms = [(feature_names[i], query_tf_idf[i]) for i in range(len(feature_names)) if query_tf_idf[i] > 0]
    query_terms.sort(key=lambda x: x[1], reverse=True)
    print("Termes les plus importants:", query_terms[:5])

    corpus = tfidf_mat[1:]

    # Recherche avec corrélation de Pearson
    results = []
    for idx, document_tf_idf in enumerate(corpus):
        pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
        print(f"Document {idx+1} - Corrélation : {pearson_corr}")
        if pearson_corr > 0.05:
            doc = documents[idx]
            results.append({
                'id': doc['id'],
                'document': doc['content'][:200] + "...",
                'similarity': round(pearson_corr * 100, 2)
            })
    
    print(f"Nombre de résultats trouvés : {len(results)}")
    
    # Trier les résultats par similarité
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return render_template('index.html', 
                         results=results,
                         query=query,
                         no_results_message="Aucun résultat trouvé pour votre recherche." if not results else None)

@app.route('/add_document', methods=['POST'])
def add_document():
    try:
        title = request.form['title']
        content = request.form['content']
        
        # Créer un nouveau document
        new_doc = {
            'id': len(documents) + 1,
            'title': title,
            'content': content
        }
        
        # Ajouter à la liste des documents
        documents.append(new_doc)
        
        # Sauvegarder dans le fichier JSON
        with open('documents.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'Document ajouté avec succès'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
