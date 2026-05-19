from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import pymysql
from django.core.files.storage import FileSystemStorage
from datetime import date
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import boto3

global uname, tokenizer, retriever, model
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def GenerationAction(request):
    if request.method == 'POST':
        query = request.POST.get('t1')

        if not query:
            return render(request, 'Generation.html', {'data': 'Please enter input text'})

        print("Query:", query)

        inputs = tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Encode question
        question_hidden_states = model.question_encoder(input_ids)[0]

        # ✅ FIX: move to CPU before numpy
        docs_dict = retriever(
            input_ids.cpu().numpy(),
            question_hidden_states.detach().cpu().numpy(),
            return_tensors="pt"
        )

        # Compute doc scores
        doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1),
            docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)

        # Generate answer
        generated = model.generate(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            max_length=50
        )

        generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        print("Generated:", generated_string)

        context = {
            'data': f"Input Text = {query}<br/><br/>Generated Text = {generated_string}"
        }

        return render(request, 'Generation.html', context)

    return render(request, 'Generation.html')

def Generation(request):
    if request.method == 'GET':
       return render(request, 'Generation.html', {})

def Retrieval(request):
    if request.method == 'GET':
       return render(request, 'Retrieval.html', {})

def DownloadFile(request):
    if request.method == 'GET':
        name = request.GET.get('name', False)
        with open('RagApp/static/files/'+name, "rb") as file:
            data = file.read()
        file.close()
        response = HttpResponse(data,content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename='+name
        return response         
    
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from django.shortcuts import render

def RetrievalAction(request):
    if request.method == 'POST':
        query = request.POST.get('t1', False)
        query = query.strip().lower()

        names = []
        search = []
        rag = []

        for root, dirs, files in os.walk('RagApp/static/files'):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                try:
                    # Read file safely
                    with open(file_path, "rb") as file:
                        raw_data = file.read()

                    #  FIX: safe decoding
                    try:
                        data = raw_data.decode('utf-8')
                    except UnicodeDecodeError:
                        data = raw_data.decode('latin-1', errors='ignore')

                    # Clean text
                    data = data.strip().lower()

                    if len(data) > 2500:
                        data = data[:2500]

                    # Store filename
                    names.append(file_name)

                    # Tokenization + embedding
                    inputs = tokenizer(data, return_tensors="pt")
                    input_ids = inputs["input_ids"]

                    question_hidden_states = model.question_encoder(input_ids)[0]
                    question_hidden_states = question_hidden_states.detach().numpy().ravel()

                    rag.append(question_hidden_states)

                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
                    continue

        # Convert to numpy
        rag = np.asarray(rag)

        # Process query
        inputs = tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]

        query_vec = model.question_encoder(input_ids)[0]
        query_vec = query_vec.detach().numpy().ravel()

        # Similarity calculation
        for i in range(len(rag)):
            try:
                score = dot(rag[i], query_vec) / (norm(rag[i]) * norm(query_vec))
                if score > 0.50:
                    search.append([names[i], score])
            except:
                continue

        # Sort results
        search.sort(key=lambda x: x[1], reverse=True)

        # HTML result
        result = "<table border=1 align=center><tr><th>Searched File Name</th><th>Retrieval Accuracy</th><th>Download File</th></tr>"

        for item in search:
            result += f"<tr><td><font size=3 color=black>{item[0]}</font></td>"
            result += f"<td><font size=3 color=black>{item[1]}</font></td>"
            result += f"<td><a href='DownloadFile?name={item[0]}'><font size=3 color=black>Download File</font></a></td></tr>"

        result += "</table><br/><br/><br/><br/>"

        return render(request, 'UserScreen.html', {'data': result})  
    
import os
from datetime import date
from django.shortcuts import render
import pymysql

def UploadDocumentAction(request):
    if request.method == 'POST':
        global uname

        try:
            # Get uploaded file
            myfile = request.FILES['t1']
            fname = myfile.name

            # File path
            upload_dir = "RagApp/static/files/"
            filepath = os.path.join(upload_dir, fname)

            # Create folder if not exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            # Delete old file if exists
            if os.path.exists(filepath):
                os.remove(filepath)

            # Save file locally
            with open(filepath, "wb+") as destination:
                for chunk in myfile.chunks():
                    destination.write(chunk)

            # Current date
            current_date = str(date.today())

            # Database connection
            db_connection = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='',
                database='rag',
                charset='utf8'
            )

            db_cursor = db_connection.cursor()

            # Insert record safely
            sql = "INSERT INTO documents VALUES(%s, %s, %s)"
            db_cursor.execute(sql, (uname, fname, current_date))

            db_connection.commit()

            print(db_cursor.rowcount, "Record Inserted")

            if db_cursor.rowcount == 1:
                status = "Document successfully uploaded"
            else:
                status = "Database insert failed"

        except Exception as e:
            print("Error:", e)
            status = "Error in uploading document"

        context = {'data': '<font size="3" color="blue">' + status + '</font>'}
        return render(request, 'UploadDocument.html', context)

    return render(request, 'UploadDocument.html')       

def UploadDocument(request):
    if request.method == 'GET':
       return render(request, 'UploadDocument.html', {})  

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})    

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def Contactus(request):
    if request.method == 'GET':
        name = "Ameerpet"
        output = '<iframe width="625" height="350" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="https://maps.google.com/maps?q='+name+'&amp;ie=UTF8&amp;&amp;output=embed"></iframe><br/>'
        context= {'data1':output}
        return render(request, 'Contactus.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'rag',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'rag',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup process completed"
        context= {'data': '<font size="3" color="blue">'+status+'</font>'}
        return render(request, 'Register.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'rag',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)  


    
        
