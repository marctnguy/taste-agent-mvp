# Sector Research — Social Film Discovery & Cultural Recommendation

**Company:** Letterboxd  
**Sector:** Social film discovery / entertainment technology  
**Project:** Taste Agent for Letterboxd  
**Stage:** Capstone Round 1

---

## 1. Research Purpose

This research examines the sector in which Letterboxd operates and the market, product and data conditions relevant to AI-powered film discovery.

Letterboxd is not primarily a streaming service. Its position sits at the intersection of:

- film discovery
- social media
- cultural logging and reviewing
- recommendation systems
- subscription-supported digital communities

The research focuses on four questions:

1. What differentiates Letterboxd from other entertainment platforms?
2. How large and engaged is its user base?
3. What types of data are generated through normal Letterboxd activity?
4. Why is this environment relevant to AI-powered discovery and personalization?

---

## 2. Sector Context

Digital film discovery is distributed across several types of products.

Streaming platforms such as Netflix, Disney+ and Prime Video recommend content primarily within their own catalogues.

Film databases such as IMDb provide structured information about films, casts, crews and ratings.

Social platforms enable discussion and cultural discovery but are not primarily designed around structured film consumption.

Letterboxd occupies a different position: it combines film discovery with social interaction and explicit cultural logging.

Users can:

- mark films as watched
- rate films
- create diary entries
- write reviews
- create lists
- maintain watchlists
- follow other members
- interact with other film viewers

This creates a platform where film discovery, personal cultural history and social activity coexist.

---

## 3. Letterboxd's Position

Letterboxd describes itself as a global social platform for film discovery and discussion.

Its product is built around individual films, but the experience extends beyond a conventional film database.

A user can maintain a persistent record of their relationship with cinema through:

- viewing history
- ratings
- diary activity
- reviews
- lists
- likes
- watchlists
- social interactions

This gives Letterboxd a distinctive position between a film database, a social network and a personal cultural diary.

Its value proposition is therefore not simply access to information about films.

It also provides users with tools to:

- record what they watch
- express opinions
- discover films
- understand their own viewing history
- see what other people are watching
- participate in film culture

This makes discovery and personalization particularly relevant product functions.

---

## 4. Letterboxd's Growth

Letterboxd has experienced substantial membership growth.

According to Tiny's 2023 acquisition announcement, Letterboxd had surpassed 10 million members across more than 200 countries at the time of the acquisition.

Tiny also reported:

- 1.8 million members in 2020
- 4.1 million members in 2021
- more than 10 million members in 2023

By Q2 2026, Tiny reported that Letterboxd had reached 30.7 million members.

This represented:

- 43% year-over-year growth
- 185% growth since Tiny's acquisition

### Reported Membership Growth

| Period | Reported Members |
|---|---:|
| 2020 | 1.8M |
| 2021 | 4.1M |
| 2023 | 10M+ |
| Q2 2026 | 30.7M |

The figures show Letterboxd's transition from a comparatively niche film community into a large global cultural platform.

---

## 5. Platform Engagement

Membership alone does not describe the scale of activity available on Letterboxd.

The platform generates large volumes of structured cultural interactions.

Letterboxd's 2025 Year in Review reported:

| 2025 Activity | Volume |
|---|---:|
| Films marked watched | 898.5M |
| Ratings | 672.5M |
| Diary entries | 332.8M |
| Reviews | 143.6M |
| Lists | 12.9M |
| Comments | 11.0M |

Letterboxd also reported approximately 652 million hours of film viewing represented by activity logged during the year.

These interactions create different types of signals.

For example:

| User Activity | Potential Information |
|---|---|
| Watched film | Consumption |
| Rating | Explicit preference |
| Like | Positive preference |
| Diary entry | Consumption + time |
| Review | Written opinion |
| Watchlist | Future interest |
| List | Curation / grouping |
| Social activity | Community interaction |

This combination is particularly relevant to personalization because the platform records more than consumption alone.

---

## 6. Explicit Preference Data

One important characteristic of Letterboxd is its use of explicit ratings.

A viewing event can show that a user consumed a film, but consumption alone does not establish whether they enjoyed it.

Ratings provide an additional explicit preference signal.

For example:

    User watched Film A
    ↓
    Consumption signal

    User watched Film A + rated it 4.5 stars
    ↓
    Consumption + explicit preference signal

At platform scale, this creates a large dataset connecting individual cultural works with expressed user preference.

For AI-powered discovery, this provides a potential basis for identifying patterns associated with what users rate positively or negatively.

---

## 7. Longitudinal Cultural Data

Letterboxd activity can also be longitudinal.

A long-term user can accumulate years of:

- watched films
- ratings
- diary entries
- reviews
- lists
- likes
- watchlist activity

This differs from a single recommendation interaction because it creates a historical record of cultural consumption.

The time dimension could support analysis of:

- stable preferences
- recent preferences
- changing interests
- viewing periods
- long-term taste evolution

Letterboxd therefore contains both preference information and temporal information relevant to personalization.

---

## 8. Existing Discovery Experience

Discovery is already central to the Letterboxd product.

Users can discover films through:

- search
- ratings
- reviews
- lists
- member activity
- social following
- watchlists
- popularity
- film pages
- filters
- streaming availability

This means an AI discovery system would not introduce film discovery as a new behaviour.

Instead, it would operate within an existing product environment where users already use Letterboxd to decide what to watch.

The relevant product question is therefore not:

> Can AI create film discovery?

It is:

> Can AI add a useful new layer to Letterboxd's existing discovery experience?

---

## 9. Existing Film Metadata

Conventional film discovery relies heavily on structured metadata.

Common attributes include:

- title
- genre
- director
- cast
- release year
- country
- runtime
- production company
- language
- popularity
- community rating

This information is highly useful for search, filtering and similarity.

For example, a user can search for:

- horror films
- films from the 1990s
- films by a specific director
- films under two hours
- highly rated films
- films available on a particular streaming service

These attributes primarily describe **what a film is**.

---

## 10. Semantic Characteristics

Cultural preference can also depend on characteristics that are less consistently represented by conventional metadata.

Examples include whether a work is:

- contemplative
- melancholic
- atmospheric
- character-driven
- experimental
- sentimental
- intimate
- playful
- unsettling
- emotionally intense

Two films can share the same genre, release period or country while differing substantially across these characteristics.

For example:

    Conventional metadata

    Genre: Drama
    Year: 2023
    Runtime: 105 minutes

does not necessarily describe whether the film is:

    contemplative
    melancholic
    intimate
    experimental
    plot-driven
    fast-paced

This creates a relevant area for semantic AI.

Large Language Models can process unstructured descriptions and other textual metadata and transform them into structured classifications.

---

## 11. From Content Metadata to Semantic Representation

A semantic classification layer can sit between conventional content metadata and personalization.

A simplified architecture is:

    FILM
      ↓
    STRUCTURED + UNSTRUCTURED METADATA
      ↓
    AI SEMANTIC CLASSIFICATION
      ↓
    STRUCTURED SEMANTIC VECTOR

Instead of replacing existing metadata, semantic classification adds another representation of the work.

For example:

| Conventional Metadata | Semantic Metadata |
|---|---|
| Drama | Character-driven |
| 2023 | Contemplative |
| 105 min | Melancholic |
| United States | Intimate |
| Director | Atmospheric |
| Cast | Emotionally intense |

The two forms of metadata answer different questions.

Conventional metadata helps describe **what the film is**.

Semantic metadata can help describe **what the film feels like or how it is narratively and stylistically constructed**.

---

## 12. Recommendation Systems and Semantic AI

Recommendation systems traditionally use several approaches.

### Content-Based Recommendation

Recommendations are generated using characteristics of items a user has previously preferred.

For films, this can include:

- genres
- actors
- directors
- keywords
- production information

### Collaborative Filtering

Recommendations are derived from behavioural similarity between users.

Conceptually:

> Users who liked similar films to you also liked this film.

### Popularity and Trending Signals

Content can be surfaced based on:

- overall popularity
- recent activity
- community ratings
- trending behaviour

### Social Discovery

On a social platform such as Letterboxd, discovery can also come from:

- friends
- followed accounts
- reviews
- lists
- community activity

### Semantic Recommendation

Modern language models and embeddings create another potential layer.

Instead of representing works only through conventional categories, content can be represented through richer semantic characteristics.

This does not require replacing existing recommendation methods.

Semantic information can complement them.

---

## 13. Why Semantic AI Is Relevant to Letterboxd

Letterboxd combines three characteristics relevant to semantic recommendation.

### Rich Content Context

Films have substantial textual and structured metadata available for semantic analysis.

### Explicit Preference Signals

Ratings provide direct evidence of whether users responded positively or negatively to individual works.

### Discovery-Oriented User Behaviour

Users already use Letterboxd to search for, evaluate and discover films.

Together, these create the possibility of connecting:

    WHAT A FILM IS LIKE
              +
    HOW A USER RATED IT
              ↓
    SEMANTIC PREFERENCE PATTERNS

For example, instead of only identifying that a user frequently watches dramas, a system could investigate whether higher ratings are associated with characteristics such as:

- contemplative pacing
- character-driven narratives
- melancholic tone
- intimate relationships
- experimental style

This is the sector and data context behind the Taste Agent project.

---

## 14. Subscription Business Model

Letterboxd operates a freemium membership model.

Users can access the platform for free, while additional functionality is available through paid Pro and Patron memberships.

At the time of this research, standard web pricing is:

| Tier | Annual Price |
|---|---:|
| Free | $0 |
| Pro | $19 |
| Patron | $49 |

Letterboxd states that membership fees are its chief source of income.

Paid membership already includes personalization and discovery-adjacent functionality such as:

- personalized statistics
- advanced filtering
- streaming-service filtering
- additional profile customization
- expanded platform features

This is relevant to the capstone because AI-powered personalization would enter a product that already monetizes enhanced user functionality through subscriptions.

Any specific Taste Agent pricing or tier placement proposed elsewhere in this project is hypothetical and does not represent current Letterboxd packaging.

---

## 15. Company Context

In 2023, Tiny acquired a majority stake in Letterboxd.

The acquisition announcement described Letterboxd as a global social platform for film discovery and discussion.

At the time, Letterboxd had more than 10 million registered members.

Tiny's investment provides relevant company context because Letterboxd is operating at substantially greater scale than during its earlier independent growth period.

By Q2 2026, reported membership had increased to 30.7 million.

This combination of:

- rapid membership growth
- large-scale behavioural data
- a subscription business model
- an established discovery use case

creates the business context in which additional personalization capabilities can be investigated.

---

## 16. Cultural Data Beyond Film

The project also considers whether cultural preference can be represented across more than one medium.

Potential sources include:

### Letterboxd

Film behaviour can provide:

- explicit ratings
- consumption history
- likes
- diary history
- reviews
- watchlists

### Goodreads

Book data can provide:

- reading history
- explicit ratings
- titles
- authors
- book metadata

Books are particularly relevant to an initial cross-media experiment because films and books share several narrative and thematic characteristics.

### Spotify

Music listening history can provide behavioural signals such as:

- listening frequency
- repeat listening
- listening duration
- skips
- recency
- long-term listening patterns

Unlike Letterboxd and Goodreads ratings, these are primarily implicit behavioural signals.

These sources represent different types of preference evidence rather than interchangeable datasets.

---

## 17. Shared and Media-Specific Semantics

Cross-media modelling requires distinguishing between characteristics that transfer across cultural media and those that do not.

Some characteristics can plausibly exist across films, books and music:

- melancholic
- atmospheric
- experimental
- nostalgic
- romantic
- playful
- high-energy

Other characteristics are specifically narrative:

- plot-driven
- character-driven
- anti-hero
- ensemble
- nonlinear narrative

Music may also require its own characteristics that do not apply naturally to films or books.

A cross-media semantic architecture can therefore be represented as:

    CULTURAL TASTE MODEL
              │
              ├── Shared semantic dimensions
              │
              ├── Film-specific dimensions
              │
              ├── Book-specific dimensions
              │
              └── Music-specific dimensions

This provides a more appropriate conceptual model than forcing every medium into an identical taxonomy.

---

## 18. Relevant Data Sources for the Project

The capstone uses or considers the following data sources.

| Source | Data Type | Project Use |
|---|---|---|
| Letterboxd export | Ratings and film history | Preference signals |
| TMDB | Film metadata | Film enrichment |
| Google Books | Book metadata | Book enrichment |
| Goodreads export | Reading and rating history | Cross-media preference signals |
| Spotify Extended Streaming History | Listening behaviour | Future cross-media preference signals |
| Tiny investor publications | Company and membership data | Business research |
| Letterboxd public publications | Platform activity and product information | Sector research |

For the formal Round 1 POC, public, synthetic or mock data is used for demonstration rather than publishing private user histories.

---

## 19. Sector Research Conclusions

The research identifies five characteristics of the Letterboxd environment that are particularly relevant to the capstone.

### 1. Discovery is a core Letterboxd behaviour

Users already use the platform to find, evaluate and discuss films.

### 2. Letterboxd contains explicit preference data

Ratings provide information beyond simple consumption history.

### 3. The platform contains longitudinal data

Viewing and rating histories can represent cultural behaviour over time.

### 4. Conventional and semantic metadata are complementary

Existing metadata describes films effectively, while semantic classification can potentially represent additional stylistic, emotional and narrative characteristics.

### 5. Letterboxd has an existing subscription model

Advanced personalization can therefore be investigated within an established paid-product context rather than requiring an entirely new monetization model.

These findings provide the research foundation for evaluating potential AI use cases for Letterboxd in the next stage of the project.

---

## Sources

- Tiny Ltd. — *Tiny Announces Majority Acquisition of Letterboxd* (2023). Used for acquisition context, historical membership figures and company positioning.
- Tiny Ltd. — *Tiny Reports Q2 2026 Results* (2026). Used for current reported membership and growth figures.
- Letterboxd — *2025 Year in Review*. Used for annual platform activity figures.
- Letterboxd — *Paid subscriptions*. Used for current membership structure and features.
- Letterboxd — *Upgrade to Letterboxd Pro*. Used for standard web subscription pricing.
- Letterboxd — *Frequently Asked Questions*. Used for information about membership revenue and platform operation.
- Letterboxd — *Purpose*. Used for platform positioning, community context and user data principles.
- TMDB — Public film metadata service used by the Round 1 POC.
- Google Books — Public book metadata service used by the Round 1 POC.
