# Analysis of LLM API Contract Violations Research Paper

## Key Findings

The research paper presents several key findings regarding LLM contract violations:

1. **Most Frequent Contracts Needed**: The study identifies that the most commonly needed contracts for ML APIs involve checking constraints on single arguments of an API or the order of API calls. These findings suggest that software engineering (SE) community could employ existing contract mining approaches to mine these contracts, enhancing the understanding of ML APIs.

2. **ML-Specific Contracts**: It was found that ML APIs require ML type checking contracts and show inter-dependency between behavioral and temporal contracts. This indicates a need for contract languages and type checking tools to include expressiveness for these additional types of contracts seen in ML APIs.

3. **Primary Root Cause of Violations**: Supplying unacceptable input values to APIs is the primary root cause behind contract violation in ML, indicating a need for better documentation and possibly runtime assertion checking tools to help avoid such violations.

4. **Impact of Violations**: On average, 56.93% of the contract violations for the ML libraries lead to a crash, highlighting the severity of contract violations in affecting the reliability of ML software.

5. **Difficulty in Contract Comprehension**: The study reveals that temporal method orders, especially "eventually" constraints, require a higher level of expertise and a longer average time to resolve, suggesting that these contracts are particularly challenging for ML API users.

6. **Localization of Contract Violations**: A significant portion of contract violations occurs during the data preprocessing and model construction stages, indicating that errors in early pipeline stages can propagate and affect subsequent stages.

## Methodology

The research employed an empirical study of posts on Stack Overflow discussing the four most often-discussed ML libraries: TensorFlow, Scikit-learn, Keras, and PyTorch. The study extracted 413 informal (English) API specifications and used these specifications to understand the root causes and effects behind ML contract violations, common patterns of ML contract violations, challenges in understanding ML contracts, and the potential for detecting violations at the API level early in the ML pipeline.

## Contract Definition

In the context of LLM APIs, contracts are defined similarly to the design by contract methodology. A contract specifies the correct usage of an API, and an incorrect usage is considered a contract violation. Contracts help document APIs and aid API users in writing correct code by specifying expected behavior, such as requiring certain methods to be called in order or setting specific argument types or values.

## Violation Criteria

Contract violations are determined based on the deviation from the specified behavior in the contract. This includes providing unacceptable input types or values, failing to maintain the required order of API calls, or not adhering to the specified preconditions and postconditions for API methods. The study identifies these violations through an analysis of Stack Overflow posts where API users discuss issues and solutions related to ML API usage.

## Impact Assessment

The impact of contract violations is measured by analyzing the effects of these violations as reported in Stack Overflow discussions. The study categorizes the effects into several types, including crashes, incorrect functionality, and bad performance. The frequency and severity of these effects are used to assess the impact of contract violations on the reliability and performance of ML software. Additionally, the study considers the difficulty in resolving contract violations and the stages in the ML pipeline where violations commonly occur to further understand the impact on ML software development.