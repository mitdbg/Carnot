# Strategy for Query 14: find me Epistemology books (No Truncation)
def execute_query(retrieve):
    # Broad topical retrievals (use large k for intersections)
    epistemology = retrieve("epistemology", 500)
    theory_of_knowledge = retrieve("theory of knowledge", 500)
    epistemological = retrieve("epistemological", 500)
    knowledge_philosophy = retrieve("knowledge (philosophy)", 500)
    intro_epistemology_phrase = retrieve("introduction to epistemology", 500)
    
    topics = epistemology | theory_of_knowledge | epistemological | knowledge_philosophy | intro_epistemology_phrase
    
    # Book-format / publication-type indicators (use large k because we'll intersect with topics)
    books_general = retrieve("book", 500)
    textbooks = retrieve("textbook", 500)
    monographs = retrieve("monograph", 500)
    handbooks = retrieve("handbook", 500)
    edition = retrieve("edition", 500)
    
    book_indicators = books_general | textbooks | monographs | handbooks | edition
    
    # Specific queries likely to match book records (union; k can be moderate)
    epistemology_books_phrase = retrieve("epistemology book", 200)
    intro_books = retrieve("introductory epistemology book", 200)
    handbook_epistemology = retrieve("handbook of epistemology", 200)
    epistemology_textbook = retrieve("epistemology textbook", 200)
    
    specific_book_matches = epistemology_books_phrase | intro_books | handbook_epistemology | epistemology_textbook
    
    # Known-title/author hints (union)
    known_titles_authors = retrieve("Gettier problem book", 200) | retrieve("reliabilism book", 200) | retrieve("Alvin Goldman epistemology", 200) | retrieve("Edwin Lehrer epistemology", 200)
    
    # Exclude likely non-book items (articles, proceedings) — use large k for difference
    journal_articles = retrieve("journal article", 500)
    conference_papers = retrieve("conference paper", 500)
    working_papers = retrieve("working paper", 500)
    
    non_book_docs = journal_articles | conference_papers | working_papers
    
    # Combine: require topical match AND book indicators, plus explicit book matches and known-title hits; then exclude non-books
    candidates = (topics & book_indicators) | specific_book_matches | known_titles_authors
    final_set = candidates - non_book_docs
    
    # Return as a list (do not truncate)
    result = list(final_set)
    return result
