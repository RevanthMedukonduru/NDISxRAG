```python
# Create an environment
conda create --name NDIS python=3.10

# Install Django
pip install django

# Create a project
django-admin startproject NDISxRAG

# Create an app
cd NDISxRAG
python manage.py startapp NDIS

# Make some changes related to url_patterns in NDISxRAG/urls.py

# Run the server
python manage.py runserver

# Decouple
pip install python-decouple
```

```python
# For FE
pip install streamlit
```