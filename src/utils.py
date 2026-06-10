class AddTokens():
  def __init__(self):
    self.new_tokens = {
      'query_begin': '<QRY>',
      'query_end': '</QRY>',
      'title_begin': '<TLE>',
      'title_end': '</TLE>',
      'text_begin': '<TXT>',
      'text_end': '</TXT>'
    }

  def add_query_tokens(self, query):
    return self.new_tokens['query_begin'] + query + self.new_tokens['query_end']
  
  def add_title_tokens(self, title):
    return self.new_tokens['title_begin'] + title + self.new_tokens['title_end']
  
  def add_text_tokens(self, text):
    return self.new_tokens['text_begin'] + text + self.new_tokens['text_end']