# ML API Contract Taxonomy Mapping

## Current Implementation vs. Paper Findings

### 1. Single API Method (SAM) Contracts

#### Paper Findings
- Most frequent contract type (28.4% of violations)
- Primary focus on argument constraints and type checking
- Includes ML-specific type checking requirements
- Root cause: Unacceptable input values most common

#### Our Implementation
- Covered in `categorization.yaml` under `Single_API_Method`
- Includes Data_Type and Boolean_Expression_Type
- **Gaps**:
  - Need more emphasis on ML-specific type checking
  - Missing explicit input validation categories
  - Could expand error handling for invalid inputs

### 2. API Method Order (AMO) Contracts

#### Paper Findings
- Second most common contract type
- Emphasis on "eventually" constraints
- Higher expertise required for temporal contracts
- Often coupled with behavioral contracts

#### Our Implementation
- Covered under `API_Method_Order` with Temporal_Order
- **Gaps**:
  - Need better representation of "eventually" constraints
  - Missing coupling patterns with behavioral contracts
  - Could add more ML-specific ordering examples

### 3. Hybrid Contracts

#### Paper Findings
- Shows interdependency between behavioral and temporal contracts
- Unique to ML APIs - allows choice between temporal ordering or state change
- Complex to understand and implement

#### Our Implementation
- Basic coverage in `Hybrid` category
- **Gaps**:
  - Need more detailed subcategories for ML-specific hybrid patterns
  - Missing documentation of choice patterns
  - Could expand interdependency examples

### 4. LLM-Specific Contracts

#### Paper Findings
- Requires specialized type checking
- Focus on input validation and preprocessing
- Early pipeline stage violations most critical

#### Our Implementation
- Extensive coverage in `LLM_Specific` category
- Good coverage of Input/Processing/Output contracts
- **Gaps**:
  - Could enhance preprocessing contract specifications
  - Need more emphasis on pipeline stage validation
  - Missing some error propagation patterns

### 5. Error Handling and Effects

#### Paper Findings
- 56.93% of violations lead to crashes
- Inadequate error messages common
- Effects propagate through ML pipeline

#### Our Implementation
- Covered under `Error_Handling`
- **Gaps**:
  - Need better categorization of error propagation
  - Could add severity impact patterns
  - Missing pipeline stage effect tracking

## Recommendations

1. **Enhance Type Checking**:
   - Add ML-specific type validation categories
   - Include tensor shape and dimension contracts
   - Add data preprocessing validation

2. **Improve Temporal Contracts**:
   - Expand "eventually" constraint patterns
   - Add more examples of correct ordering
   - Include pipeline stage dependencies

3. **Strengthen Hybrid Categories**:
   - Add more ML-specific hybrid patterns
   - Document state-ordering choice patterns
   - Include real-world examples

4. **Enhance Error Handling**:
   - Add error propagation patterns
   - Include severity assessment guidelines
   - Add pipeline stage impact tracking

5. **Add Pipeline Stage Context**:
   - Tag contracts with pipeline stages
   - Add preprocessing-specific contracts
   - Include stage dependency patterns 