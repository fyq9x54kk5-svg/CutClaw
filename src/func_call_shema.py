# Adapted from https://github.com/peterroelants/annotated-docs
"""
Function Call Schema Generator - Convert Python functions to JSON Schema for LLM function calling.
函数调用模式生成器 - 将 Python 函数转换为 JSON Schema 以供 LLM 函数调用。

This module provides utilities to automatically generate JSON schemas from Python function
signatures and docstrings. These schemas are used by LLMs to understand available tools
and their parameters for function calling (tool use).
此模块提供实用工具，从 Python 函数签名和文档字符串自动生成 JSON 模式。
这些模式被 LLM 用于理解可用的工具及其参数，以进行函数调用（工具使用）。

Key features:
主要功能：
1. Extract parameter types and descriptions from function annotations
1. 从函数注解中提取参数类型和描述
2. Generate Pydantic models dynamically for validation
2. 动态生成 Pydantic 模型进行验证
3. Produce clean JSON schemas without unnecessary titles
3. 生成不含多余标题的简洁 JSON 模式

Java 类比：类似 Java 反射 + Jackson JSON Schema 生成器的组合，但更加简洁和自动化。
"""

import inspect
from collections.abc import Callable
from typing import Any, Final, Required, TypedDict, TypeVar

import pydantic
import pydantic.json_schema

# JSON Schema 中返回值的关键字
RETURNS_KEY: Final[str] = "returns"

# 泛型类型变量，用于类型提示
T = TypeVar("T")


class FunctionJSONSchema(TypedDict, total=False):
    """
    TypedDict defining the structure of a function's JSON schema.
    定义函数 JSON 模式结构的 TypedDict。
    
    This schema is used by LLMs to understand function signatures.
    此模式被 LLM 用于理解函数签名。
    
    Java 类比：类似一个 POJO 类，定义了 name、description、parameters 三个字段。
    """
    name: Required[str]  # 函数名称（必需）
    description: str  # 函数描述（可选）
    parameters: dict[str, Any]  # 参数模式（可选）


def as_json_schema(func: Callable) -> FunctionJSONSchema:
    """
    Return a JSON schema for the given function.
    返回给定函数的 JSON 模式。
    
    This is the main entry point for converting a Python function into a JSON schema
    that can be used by LLMs for function calling. It extracts:
    这是将 Python 函数转换为 LLM 可用于函数调用的 JSON 模式的主要入口点。它提取：
    - Function name from __name__
    - 从 __name__ 提取函数名称
    - Description from docstring
    - 从文档字符串提取描述
    - Parameter types and descriptions from annotations
    - 从注解提取参数类型和描述
    
    Args:
        func: 要转换的 Python 函数 (Python function to convert)
    Returns:
        包含 name、description、parameters 的字典 (Dict with name, description, parameters)
    
    Example usage:
    使用示例：
    >>> def add(a: int, b: int) -> int:
    ...     """Add two numbers."""
    ...     return a + b
    >>> schema = as_json_schema(add)
    >>> print(schema['name'])  # 'add'
    """
    # 获取参数的 JSON 模式
    parameters_schema = get_parameters_schema(func)
    # 从文档字符串中提取描述
    description = ""
    if func.__doc__:
        # inspect.cleandoc(): 清理文档字符串的缩进
        # Java 类比：类似 String.stripIndent()（Java 15+）
        description = inspect.cleandoc(func.__doc__).strip()
    schema_dct: FunctionJSONSchema = {
        "name": func.__name__,
        "description": description,
        "parameters": parameters_schema,
    }
    return schema_dct


def doc(description) -> Any:
    """
    Annotate a variable with a description.
    为变量添加描述注解。
    
    This is a helper function used with Python's typing.Annotated to provide
    descriptions for function parameters. It wraps pydantic.Field().
    这是一个辅助函数，与 Python 的 typing.Annotated 一起使用，为函数参数提供描述。
    它包装了 pydantic.Field()。
    
    Args:
        description: 参数的描述文本 (Description text for the parameter)
    Returns:
        Pydantic Field 对象，包含描述信息 (Pydantic Field object with description)
    
    Example usage:
    使用示例：
    >>> from typing import Annotated
    >>> def greet(name: Annotated[str, doc("The person's name")]):
    ...     pass
    """
    return pydantic.Field(description=description)


def get_parameters_schema(func: Callable) -> dict[str, Any]:
    """
    Return a JSON schema for the parameters of the given function.
    返回给定函数参数的 JSON 模式。
    
    This function creates a Pydantic model from the function's parameters and
    then converts it to a JSON schema. The schema includes:
    此函数从函数的参数创建 Pydantic 模型，然后将其转换为 JSON 模式。模式包括：
    - Parameter names and types
    - 参数名称和类型
    - Required/optional status
    - 必需/可选状态
    - Descriptions (if provided via doc())
    - 描述（如果通过 doc() 提供）
    
    Args:
        func: 要提取参数模式的函数 (Function to extract parameter schema from)
    Returns:
        JSON 模式字典，包含 type、properties、required 等字段 (JSON schema dict)
    """
    # 获取参数的 Pydantic 模型
    parameter_model = get_parameter_model(func)
    # 将 Pydantic 模型转换为 JSON 模式
    return parameter_model.model_json_schema(
        schema_generator=GenerateJsonSchemaNoTitle,
        mode="validation",
    )


def get_parameter_model(func: Callable) -> pydantic.BaseModel:
    """
    Return a Pydantic model for the parameters of the given function.
    返回给定函数参数的 Pydantic 模型。
    
    This function dynamically creates a Pydantic BaseModel class from the function's
    signature. Each parameter becomes a field in the model with its type annotation
    and default value (if any).
    此函数从函数签名动态创建 Pydantic BaseModel 类。每个参数都成为模型中的一个字段，
    包含其类型注解和默认值（如果有）。
    
    Java 类比：类似使用反射获取方法参数，然后动态创建一个 Java Bean 类。
    
    Args:
        func: 要提取参数的函数 (Function to extract parameters from)
    Returns:
        动态创建的 Pydantic 模型类 (Dynamically created Pydantic model class)
    Raises:
        ValueError: 如果参数没有类型注解 (If parameter has no type annotation)
    """
    # 字段定义字典：{参数名: (类型, 默认值)}
    field_definitions: dict[str, tuple[Any, Any]] = {}
    # inspect.signature(): 获取函数签名对象
    # Java 类比：Method.getParameters()
    for name, obj in inspect.signature(func).parameters.items():
        # 检查是否有类型注解
        if obj.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"`{func.__name__}` parameter `{name!s}` has no annotation, please provide an notation to be able to generate the function specification."
            )
        # 检查是否有默认值
        if obj.default == inspect.Parameter.empty:
            # 没有默认值：必需参数，使用 pydantic.Field(...) 标记为必需
            # Java 类比：没有默认值的构造函数参数
            field_definitions[name] = (obj.annotation, pydantic.Field(...))
        else:
            # 有默认值：可选参数
            # Java 类比：有默认值的构造函数参数或 setter 方法
            field_definitions[name] = (obj.annotation, obj.default)
    _model_name = ""  # Empty model name
    # pydantic.create_model(): 动态创建 Pydantic 模型类
    # Java 类比：类似使用字节码操作库（如 ASM 或 Javassist）动态生成类
    return pydantic.create_model(_model_name, **field_definitions)  # type: ignore


def get_returns_schema(func: Callable) -> dict[str, Any]:
    """
    Return a JSON schema for the return value of the given function.
    返回给定函数返回值的 JSON 模式。
    
    This extracts the return type annotation and converts it to a JSON schema.
    The result is flattened to remove the wrapper object structure.
    此函数提取返回类型注解并将其转换为 JSON 模式。结果被扁平化以移除包装对象结构。
    
    Args:
        func: 要提取返回值模式的函数 (Function to extract return schema from)
    Returns:
        返回值的 JSON 模式字典 (JSON schema dict for return value)
    """
    # 获取返回值的 Pydantic 模型
    returns_model = get_returns_model(func)
    # 将模型转换为 JSON 模式
    return_schema = returns_model.model_json_schema(
        schema_generator=GenerateJsonSchemaNoTitle,
        mode="validation",
    )
    # 提取 properties 并扁平化，移除 'returns' 包装层
    properties = return_schema.pop("properties")
    # |= 运算符：字典合并（Python 3.9+）
    # Java 类比：map.putAll(otherMap)
    return_schema |= properties[RETURNS_KEY]
    # 清理不需要的字段
    if "required" in return_schema:
        del return_schema["required"]
    if "type" in return_schema and return_schema["type"] == "object":
        del return_schema["type"]
    return return_schema


def get_returns_model(func: Callable) -> pydantic.BaseModel:
    """
    Return a Pydantic model for the returns of the given function.
    返回给定函数返回值的 Pydantic 模型。
    
    This creates a dynamic Pydantic model with a single field named 'returns'
    that has the type of the function's return annotation.
    这创建一个动态的 Pydantic 模型，包含一个名为 'returns' 的字段，
    其类型为函数的返回注解类型。
    
    Args:
        func: 要提取返回类型的函数 (Function to extract return type from)
    Returns:
        动态创建的 Pydantic 模型类 (Dynamically created Pydantic model class)
    Raises:
        ValueError: 如果函数没有返回类型注解 (If function has no return type annotation)
    """
    # 获取函数的返回类型注解
    # Java 类比：Method.getGenericReturnType()
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation == inspect.Signature.empty:
        raise ValueError(
            f"`{func.__name__}` has no return annotation, please provide an annotation to be able to generate the function specification."
        )
    # 创建字段定义：{'returns': (返回类型, 必需字段)}
    field_definitions: dict[str, tuple[Any, Any]] = {
        RETURNS_KEY: (return_annotation, pydantic.Field(...))
    }
    _model_name = ""  # Empty model name
    # 动态创建模型
    return pydantic.create_model(_model_name, **field_definitions)  # type: ignore


class GenerateJsonSchemaNoTitle(pydantic.json_schema.GenerateJsonSchema):
    """
    Custom JSON schema generator that removes 'title' fields from the output.
    自定义 JSON 模式生成器，从输出中移除 'title' 字段。
    
    Pydantic's default schema generator includes 'title' fields based on model
    and field names. For LLM function calling, these titles are unnecessary and
    can clutter the schema. This class overrides the generation methods to strip them out.
    Pydantic 的默认模式生成器会根据模型和字段名称包含 'title' 字段。
    对于 LLM 函数调用，这些标题是不必要的，会使模式变得混乱。此类重写生成方法以移除它们。
    
    Java 类比：类似继承 Jackson 的 JsonSerializer 并自定义输出格式。
    """
    
    def generate(
        self, schema, mode="validation"
    ) -> pydantic.json_schema.JsonSchemaValue:
        """
        Generate JSON schema and remove top-level title.
        生成 JSON 模式并移除顶层标题。
        """
        # 调用父类的生成方法
        json_schema = super().generate(schema, mode=mode)
        # 移除顶层的 title 字段
        if "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def get_schema_from_definitions(
        self, json_ref
    ) -> pydantic.json_schema.JsonSchemaValue | None:
        """
        Get schema from definitions and remove title if present.
        从定义中获取模式，如果存在则移除 title。
        """
        json_schema = super().get_schema_from_definitions(json_ref)
        if json_schema and "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def field_title_should_be_set(self, schema) -> bool:
        """
        Override to never set field titles.
        重写此方法以从不设置字段标题。
        
        Returns:
            始终返回 False (Always returns False)
        """
        return False