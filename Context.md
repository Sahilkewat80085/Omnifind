# ROLE

You are a Senior Software Architect and Full Stack AI Engineer.

Your task is to build the first 20% MVP of my Final Year Project.

Project Name:

OmniFind
AI-Based Context-Aware Retrieval System for Heterogeneous Digital Assets

Do NOT build unnecessary features.

Focus on writing clean, scalable and production-ready code.

Always explain your architectural decisions before writing code.

Never generate everything in one huge response.
Implement one module at a time.

------------------------------------------------------------

# PROJECT OVERVIEW

OmniFind is an AI-powered desktop application that performs semantic search across documents stored on a user's local computer.

Unlike Windows Search, OmniFind does not rely only on keyword matching.

Instead, it converts documents into vector embeddings and allows users to search using natural language.

For this milestone, the application does NOT generate AI answers.

It only retrieves the most semantically relevant document chunks.

This milestone is called:

Semantic Document Indexing & Retrieval Engine

------------------------------------------------------------

# TECH STACK

Desktop
Tauri

Frontend
React
TypeScript
Vite

Backend
FastAPI

Programming Language
Python

Vector Database
Qdrant

Metadata Database
SQLite

Embedding Model

BAAI/bge-small-en-v1.5

Libraries

Sentence Transformers

PyMuPDF

python-docx

LangChain (only if useful)

Pydantic

SQLAlchemy

------------------------------------------------------------

# MVP GOAL

The application should allow the user to:

Select a local folder

↓

Scan the folder recursively

↓

Read all supported files

↓

Extract text

↓

Split into chunks

↓

Generate embeddings

↓

Store vectors in Qdrant

↓

Store metadata in SQLite

↓

Perform semantic search

↓

Return the most relevant chunks with metadata

No Gemini.

No LLM.

No RAG.

No AI-generated answers.

Only semantic retrieval.

------------------------------------------------------------

# SUPPORTED FILE TYPES

Implement only

PDF

DOCX

TXT

Ignore all other formats.

------------------------------------------------------------

# EXPECTED USER FLOW

1.

Launch OmniFind Desktop App

2.

Click

Select Folder

3.

Choose

C:\Users\Sahil\Documents

4.

Click

Index Files

Application should display

Scanning...

Extracting...

Chunking...

Generating Embeddings...

Saving...

Completed

5.

Search

Example

"database normalization"

Results

DBMS.pdf

Page 18

Similarity Score

Chunk Preview

Open File Button

------------------------------------------------------------

# PROJECT STRUCTURE

Create a professional project structure.

omnifind/

frontend/

backend/

api/

core/

services/

database/

models/

schemas/

utils/

parsers/

embeddings/

vectorstore/

storage/

Do not place business logic inside API routes.

Follow Clean Architecture principles.

------------------------------------------------------------

# BACKEND MODULES

Create separate services.

FolderScanner

DocumentParser

TextChunker

EmbeddingService

VectorService

MetadataService

SearchService

Every module should have a single responsibility.

------------------------------------------------------------

# DATABASE

SQLite

Store

File ID

File Name

Path

Extension

Indexed Date

Chunk Count

------------------------------------------------------------

# VECTOR DATABASE

Qdrant

Collection

omnifind_documents

Payload

file_name

file_path

page_number

chunk_text

chunk_index

------------------------------------------------------------

# FRONTEND

React

Pages

Dashboard

Settings

Components

Sidebar

FolderPicker

ProgressBar

SearchBar

ResultCard

StatsCard

LoadingScreen

Use a clean professional UI.

------------------------------------------------------------

# SEARCH RESULTS

Display

File Name

Page Number

Similarity Score

Matched Text Chunk

Open File Button

Example

--------------------------------

DBMS.pdf

Page 18

Similarity

94%

Normalization is the process of...

[Open]

--------------------------------

------------------------------------------------------------

# UI REQUIREMENTS

Professional

Minimal

Dark Mode

Modern Cards

Responsive Layout

No fancy animations

No glassmorphism

Looks like VS Code + Notion

------------------------------------------------------------

# CODING STANDARDS

Use Python typing

Use TypeScript

Use environment variables

No hardcoded paths

No hardcoded API keys

Write reusable services

Use dependency injection where appropriate

Use proper exception handling

Create README

Create requirements.txt

Create package.json

Document every module

------------------------------------------------------------

# IMPORTANT

Do NOT implement

Gemini

Chatbot

RAG

OCR

Images

Face Detection

Object Detection

Knowledge Graph

Cloud Storage

Authentication

Continuous Folder Monitoring

These belong to future milestones.

------------------------------------------------------------

# DELIVERABLE

At the end of this milestone, the application must be able to:

✔ Launch as a desktop application

✔ Allow folder selection

✔ Scan folders recursively

✔ Parse PDF, DOCX and TXT files

✔ Extract text

✔ Chunk text

✔ Generate embeddings

✔ Store embeddings in Qdrant

✔ Store metadata in SQLite

✔ Perform semantic search

✔ Display the top matching chunks with similarity scores and source information

This is the complete deliverable for the 20% implementation.

------------------------------------------------------------

# IMPLEMENTATION STYLE

Do NOT generate the whole project at once.

Act as a senior engineer working with me.

First:

1. Design the complete architecture.
2. Explain every folder and module.
3. Show the project tree.
4. Explain how each module communicates.
5. Wait until the architecture is finalized.

Then implement one module at a time in this order:

Step 1 — Project setup (Tauri + React + FastAPI)
Step 2 — Database setup (SQLite)
Step 3 — Folder scanner
Step 4 — Document parsers
Step 5 — Text chunker
Step 6 — Embedding service
Step 7 — Qdrant integration
Step 8 — Metadata storage
Step 9 — Search API
Step 10 — React UI
Step 11 — End-to-end testing

At the end of every step:

• Explain what was built.
• Explain why it is designed that way.
• Show how to test it before moving to the next step.

Never skip directly to coding without first explaining the architecture.