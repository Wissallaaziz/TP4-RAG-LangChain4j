package ma.emsi.laaziz.rag;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

public class TestRoutage {

    public static void main(String[] args) {

        String apiKey = System.getProperty("GEMINI_API_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Erreur : propriété GEMINI_API_KEY absente.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();


        EmbeddingStore<TextSegment> ragStore =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        EmbeddingStore<TextSegment> autreStore =
                creerEmbeddingStore("autre.pdf", embeddingModel);


        ContentRetriever ragRetriever =
                creerContentRetriever(ragStore, embeddingModel);

        ContentRetriever autreRetriever =
                creerContentRetriever(autreStore, embeddingModel);

   
        Map<ContentRetriever, String> descriptions = new HashMap<>();

        descriptions.put(
                ragRetriever,
                "Support de cours sur l'intelligence artificielle, " +
                        "les modèles de langage, le fine-tuning, " +
                        "les embeddings et le RAG."
        );

        descriptions.put(
                autreRetriever,
                "Document sur l'astronomie, les planètes " +
                        "et le système solaire."
        );


        QueryRouter router =
                new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(router)
                        .build();

        ChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(memory)
                        .retrievalAugmentor(augmentor)
                        .build();

        System.out.println("Test 3 - Routage prêt.");
        System.out.println("Tapez 'exit' pour quitter.");

        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.print("\nQuestion : ");

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().trim();

                if ("exit".equalsIgnoreCase(question)) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                try {
                    String reponse = assistant.chat(question);
                    System.out.println("\nRéponse : " + reponse);
                } catch (RuntimeException e) {
                    System.err.println(
                            "Erreur pendant la requête : " +
                                    e.getMessage());
                }
            }
        }
    }


    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            String nomDocument,
            EmbeddingModel embeddingModel) {

        DocumentParser parser = new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        nomDocument, parser);

        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(embeddings, segments);

        System.out.println(
                nomDocument + " : " +
                        segments.size() + " segments enregistrés.");

        return store;
    }

  
    private static ContentRetriever creerContentRetriever(
            EmbeddingStore<TextSegment> store,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}