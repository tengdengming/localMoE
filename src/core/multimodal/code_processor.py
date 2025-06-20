"""
代码处理器 - 处理编程语言代码输入
支持多种编程语言的语法分析和特征提取
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import ast
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


@dataclass
class CodeConfig:
    """代码处理配置"""
    max_length: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = 50000
    max_ast_depth: int = 20
    enable_syntax_analysis: bool = True
    enable_semantic_analysis: bool = True
    supported_languages: List[str] = None


class CodeTokenizer:
    """代码tokenizer"""
    
    def __init__(self, config: CodeConfig):
        self.config = config
        
        # 基础token映射
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<cls>": 2,
            "<sep>": 3,
            "<mask>": 4,
            "<indent>": 5,
            "<dedent>": 6,
            "<newline>": 7
        }
        
        # 编程语言关键词
        self.keywords = {
            CodeLanguage.PYTHON: [
                "def", "class", "if", "else", "elif", "for", "while", "try", "except",
                "import", "from", "return", "yield", "lambda", "with", "as", "pass",
                "break", "continue", "and", "or", "not", "in", "is", "None", "True", "False"
            ],
            CodeLanguage.JAVASCRIPT: [
                "function", "var", "let", "const", "if", "else", "for", "while", "do",
                "switch", "case", "default", "try", "catch", "finally", "return",
                "class", "extends", "import", "export", "async", "await", "true", "false", "null"
            ],
            CodeLanguage.JAVA: [
                "public", "private", "protected", "static", "final", "class", "interface",
                "extends", "implements", "if", "else", "for", "while", "do", "switch",
                "case", "default", "try", "catch", "finally", "return", "new", "this", "super"
            ]
        }
        
        # 构建词汇表
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """构建词汇表"""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        current_id = len(self.special_tokens)
        
        # 添加关键词
        for lang_keywords in self.keywords.values():
            for keyword in lang_keywords:
                if keyword not in self.token_to_id:
                    self.token_to_id[keyword] = current_id
                    self.id_to_token[current_id] = keyword
                    current_id += 1
        
        # 添加常见操作符
        operators = [
            "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", ">", "<=", ">=",
            "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "++", "--", "+=", "-=",
            "*=", "/=", "%=", "(", ")", "[", "]", "{", "}", ".", ",", ";", ":"
        ]
        
        for op in operators:
            if op not in self.token_to_id:
                self.token_to_id[op] = current_id
                self.id_to_token[current_id] = op
                current_id += 1
        
        self.vocab_size = current_id
        logger.info(f"CodeTokenizer vocabulary size: {self.vocab_size}")
    
    def detect_language(self, code: str) -> CodeLanguage:
        """检测编程语言"""
        code_lower = code.lower()
        
        # 简单的语言检测规则
        if "def " in code or "import " in code or "from " in code:
            return CodeLanguage.PYTHON
        elif "function " in code or "var " in code or "let " in code or "const " in code:
            return CodeLanguage.JAVASCRIPT
        elif "public class " in code or "private " in code or "public static void main" in code:
            return CodeLanguage.JAVA
        elif "#include" in code or "int main(" in code:
            return CodeLanguage.CPP
        elif "func " in code or "package " in code:
            return CodeLanguage.GO
        elif "fn " in code or "let mut " in code:
            return CodeLanguage.RUST
        else:
            return CodeLanguage.UNKNOWN
    
    def tokenize_code(self, code: str) -> List[str]:
        """代码tokenization"""
        # 预处理：处理缩进
        lines = code.split('\n')
        tokens = []
        indent_stack = [0]
        
        for line in lines:
            if line.strip():  # 非空行
                # 计算缩进
                indent = len(line) - len(line.lstrip())
                
                # 处理缩进变化
                if indent > indent_stack[-1]:
                    tokens.append("<indent>")
                    indent_stack.append(indent)
                elif indent < indent_stack[-1]:
                    while indent_stack and indent < indent_stack[-1]:
                        tokens.append("<dedent>")
                        indent_stack.pop()
                
                # tokenize行内容
                line_tokens = self._tokenize_line(line.strip())
                tokens.extend(line_tokens)
            
            tokens.append("<newline>")
        
        # 处理剩余的dedent
        while len(indent_stack) > 1:
            tokens.append("<dedent>")
            indent_stack.pop()
        
        return tokens
    
    def _tokenize_line(self, line: str) -> List[str]:
        """tokenize单行代码"""
        # 简单的正则表达式tokenization
        # 匹配标识符、数字、字符串、操作符等
        pattern = r'''
            (?P<STRING>["'](?:[^"'\\]|\\.)*["']) |  # 字符串
            (?P<NUMBER>\d+\.?\d*) |                 # 数字
            (?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*) | # 标识符
            (?P<OPERATOR>[+\-*/=<>!&|^~%]+) |       # 操作符
            (?P<DELIMITER>[(){}\[\],;:.]) |         # 分隔符
            (?P<WHITESPACE>\s+)                     # 空白字符
        '''
        
        tokens = []
        for match in re.finditer(pattern, line, re.VERBOSE):
            token = match.group()
            if match.lastgroup != 'WHITESPACE':  # 忽略空白字符
                tokens.append(token)
        
        return tokens
    
    def encode(self, code: str) -> Dict[str, torch.Tensor]:
        """编码代码为token IDs"""
        tokens = self.tokenize_code(code)
        
        # 添加特殊token
        tokens = ["<cls>"] + tokens + ["<sep>"]
        
        # 截断或填充
        if len(tokens) > self.config.max_length:
            tokens = tokens[:self.config.max_length]
        
        # 转换为IDs
        input_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id["<unk>"])
            input_ids.append(token_id)
        
        # 填充
        while len(input_ids) < self.config.max_length:
            input_ids.append(self.token_to_id["<pad>"])
        
        # 创建attention mask
        attention_mask = [1 if token_id != self.token_to_id["<pad>"] else 0 for token_id in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "tokens": tokens
        }


class ASTAnalyzer:
    """AST语法分析器"""
    
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
    
    def parse_python_ast(self, code: str) -> Dict[str, Any]:
        """解析Python AST"""
        try:
            tree = ast.parse(code)
            ast_features = self._extract_ast_features(tree)
            return ast_features
        except SyntaxError as e:
            logger.warning(f"Python AST parsing failed: {e}")
            return {"error": str(e), "node_count": 0, "depth": 0}
    
    def _extract_ast_features(self, node, depth=0) -> Dict[str, Any]:
        """提取AST特征"""
        if depth > self.max_depth:
            return {"truncated": True}
        
        features = {
            "node_type": type(node).__name__,
            "depth": depth,
            "children": []
        }
        
        # 递归处理子节点
        for child in ast.iter_child_nodes(node):
            child_features = self._extract_ast_features(child, depth + 1)
            features["children"].append(child_features)
        
        return features
    
    def get_ast_statistics(self, ast_features: Dict[str, Any]) -> Dict[str, int]:
        """获取AST统计信息"""
        stats = {"node_count": 0, "max_depth": 0, "node_types": {}}
        
        def count_nodes(node_features, current_depth=0):
            if isinstance(node_features, dict):
                stats["node_count"] += 1
                stats["max_depth"] = max(stats["max_depth"], current_depth)
                
                node_type = node_features.get("node_type", "unknown")
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
                
                for child in node_features.get("children", []):
                    count_nodes(child, current_depth + 1)
        
        count_nodes(ast_features)
        return stats


class CodeProcessor(nn.Module):
    """
    代码处理器
    完整的代码处理流水线
    """
    
    def __init__(self, config: CodeConfig):
        super().__init__()
        self.config = config
        
        # 代码tokenizer
        self.tokenizer = CodeTokenizer(config)
        
        # AST分析器
        if config.enable_syntax_analysis:
            self.ast_analyzer = ASTAnalyzer(config.max_ast_depth)
        
        # 更新配置中的vocab_size
        config.vocab_size = self.tokenizer.vocab_size
        
        # 嵌入层
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        
        # 语言类型嵌入
        self.language_embeddings = nn.Embedding(len(CodeLanguage), config.hidden_size)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        
        # 特征投影层
        self.feature_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # AST特征编码器
        if config.enable_syntax_analysis:
            self.ast_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"CodeProcessor initialized with vocab_size: {config.vocab_size}")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        code: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码
            language_ids: 语言类型IDs
            code: 原始代码
            
        Returns:
            Dict[str, torch.Tensor]: 处理结果
        """
        # 如果提供了原始代码，先进行编码
        if code is not None:
            if isinstance(code, str):
                code = [code]
            
            batch_encoded = []
            batch_languages = []
            
            for code_snippet in code:
                encoded = self.tokenizer.encode(code_snippet)
                language = self.tokenizer.detect_language(code_snippet)
                
                batch_encoded.append(encoded)
                batch_languages.append(language.value)
            
            # 组合批次数据
            input_ids = torch.stack([enc["input_ids"] for enc in batch_encoded])
            attention_mask = torch.stack([enc["attention_mask"] for enc in batch_encoded])
            
            # 语言类型编码
            language_id_map = {lang.value: i for i, lang in enumerate(CodeLanguage)}
            language_ids = torch.tensor([
                language_id_map.get(lang, language_id_map["unknown"]) 
                for lang in batch_languages
            ], dtype=torch.long)
        
        if input_ids is None:
            raise ValueError("Either input_ids or code must be provided")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 词嵌入
        token_embeddings = self.embeddings(input_ids)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 语言类型嵌入
        if language_ids is not None:
            language_ids = language_ids.to(device)
            language_embeddings = self.language_embeddings(language_ids).unsqueeze(1)
            language_embeddings = language_embeddings.expand(-1, seq_len, -1)
        else:
            language_embeddings = torch.zeros_like(token_embeddings)
        
        # 组合嵌入
        embeddings = token_embeddings + position_embeddings + language_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 处理注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            # 转换为Transformer期望的格式
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer编码
        encoded = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        
        # 特征投影
        projected_features = self.feature_projection(encoded)
        
        # 池化输出
        if attention_mask is not None:
            # 使用注意力掩码进行加权平均
            mask = (attention_mask != -10000.0).float().unsqueeze(-1)
            pooled_output = (projected_features * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_output = projected_features.mean(dim=1)
        
        return {
            "last_hidden_state": encoded,
            "projected_features": projected_features,
            "pooler_output": pooled_output,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "language_ids": language_ids
        }
    
    def extract_code_features(
        self,
        code: Union[str, List[str]],
        include_ast: bool = True
    ) -> Dict[str, Any]:
        """
        提取代码特征
        
        Args:
            code: 输入代码
            include_ast: 是否包含AST特征
            
        Returns:
            Dict[str, Any]: 代码特征
        """
        with torch.no_grad():
            outputs = self.forward(code=code)
            
            features = {
                "semantic_features": outputs["pooler_output"],
                "token_features": outputs["projected_features"],
                "language_ids": outputs["language_ids"]
            }
            
            # 添加AST特征
            if include_ast and self.config.enable_syntax_analysis:
                if isinstance(code, str):
                    code = [code]
                
                ast_features = []
                for code_snippet in code:
                    language = self.tokenizer.detect_language(code_snippet)
                    if language == CodeLanguage.PYTHON:
                        ast_feature = self.ast_analyzer.parse_python_ast(code_snippet)
                        ast_stats = self.ast_analyzer.get_ast_statistics(ast_feature)
                        ast_features.append(ast_stats)
                    else:
                        ast_features.append({"node_count": 0, "max_depth": 0})
                
                features["ast_features"] = ast_features
            
            return features
    
    def get_code_similarity(self, code1: str, code2: str) -> float:
        """计算代码相似度"""
        features1 = self.extract_code_features(code1)
        features2 = self.extract_code_features(code2)
        
        # 使用余弦相似度
        feat1 = features1["semantic_features"]
        feat2 = features2["semantic_features"]
        
        similarity = torch.cosine_similarity(feat1, feat2, dim=-1)
        return similarity.item()
