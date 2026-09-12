"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Any, Dict
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    "Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau của "
                    f"khách hàng mà KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api",
                }
            except Exception:
                # Theo slide: dùng câu trả lời mẫu nếu SDK/API không khả dụng.
                pass

        return {
            "answer": (
                "Bạn có thể tìm chuyến bay trên các trang hàng không. "
                "Về thời tiết, bạn nên tra cứu trên trang dự báo thời tiết."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def parse_city_code(text: str) -> str:
        aliases = {
            "HAN": r"\b(?:HAN|HÀ NỘI)\b",
            "SGN": r"\b(?:SGN|HỒ CHÍ MINH|SÀI GÒN)\b",
            "DAD": r"\b(?:DAD|ĐÀ NẴNG)\b",
        }
        matches = []
        for code, pattern in aliases.items():
            match = re.search(pattern, text.upper())
            if match:
                matches.append((match.start(), code))
        return min(matches)[1] if matches else ""

    @staticmethod
    def execute_action(action_json: str):
        """Chuẩn hóa tên tool và chuyển lỗi JSON/tool thành Observation."""
        try:
            action = json.loads(action_json)
        except (ValueError, TypeError):
            return {"error": "Invalid JSON format"}
        if not isinstance(action, dict):
            return {"error": "Action must be a JSON object"}
        name = action.get("name")
        args = action.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return {"error": "Action requires a tool name and an args object"}
        tool = TOOL_MAP.get(name.strip().lower())
        if tool is None:
            return {"error": "Unknown tool"}
        try:
            return tool(**args)
        except Exception as exc:
            return {"error": f"Tool execution failed: {type(exc).__name__}"}

    def summarize_observations(self) -> str:
        parts = []
        for step in self.trace:
            if "action" not in step:
                continue
            action, obs = step["action"], step["observation"]
            if isinstance(obs, dict) and "error" in obs:
                parts.append(f"Không thể tra cứu: {obs['error']}")
            elif action["name"] == "get_flight_info":
                args = action["args"]
                route = f"{args['origin']} đi {args['destination']}"
                if not obs:
                    parts.append(
                        f"Không tìm thấy chuyến bay từ {route} "
                        f"trong ngân sách {args['max_price']:,} VNĐ."
                    )
                else:
                    lines = [
                        f"- {fl['airline']} ({fl['flight_number']}): "
                        f"{fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ"
                        for fl in obs
                    ]
                    parts.append(f"Chuyến bay từ {route}:\n" + "\n".join(lines))
            else:
                parts.append(
                    f"Thời tiết tại {obs['city']}: {obs['temperature_c']}°C, "
                    f"{obs['condition']}.\nGợi ý trang phục: {obs['recommendation']}"
                )
        return "\n\n".join(parts)

    def plan_and_execute_step(self, user_input: str, iteration: int):
        """Chọn hành động theo từ khóa; không cần API key cho Task 2."""
        text = user_input.lower()

        def finish(answer, thought):
            self.trace.append({
                "iteration": iteration, "thought": thought, "final_answer": answer,
            })
            return answer, True

        if "chính sách" in text or "đổi trả" in text:
            return finish(
                "Theo tình huống mẫu của lab, vé máy bay Vinpearl hỗ trợ đổi ngày "
                "trước 24 giờ so với giờ khởi hành, phí 350.000 VNĐ/vé "
                "cộng chênh lệch giá vé (nếu có).",
                "Đây là câu hỏi FAQ trong lab, không cần gọi tool.",
            )

        needs_flight = any(k in text for k in ["chuyến bay", "vé", "bay từ"])
        needs_weather = any(k in text for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa"])
        actions = []
        destination = ""
        if needs_flight:
            route = re.search(r"\btừ\s+(.+?)\s+(?:đi|đến|tới)\s+(.+)", text)
            if not route:
                route = re.search(r"\b(HAN|SGN|DAD)\s+(?:đi|đến|tới)\s+(.+)", user_input, re.I)
            origin = self.parse_city_code(route.group(1)) if route else ""
            destination = self.parse_city_code(route.group(2)) if route else ""
            if not origin or not destination:
                return finish("Bạn hãy cung cấp điểm đi và điểm đến (HAN, SGN hoặc DAD).",
                              "Chưa đủ thông tin để tra cứu chuyến bay.")
            max_price = 5000000
            budget = re.search(r"(?:dưới|tối đa|không quá)\s*([\d.,]+)\s*(triệu|k|nghìn|ngàn)?", text)
            if budget:
                amount, unit = budget.groups()
                if unit:
                    multiplier = 1000000 if unit == "triệu" else 1000
                    max_price = int(float(amount.replace(",", ".")) * multiplier)
                else:
                    max_price = int(amount.replace(".", "").replace(",", ""))
            actions.append({"name": "get_flight_info", "args": {
                "origin": origin, "destination": destination, "max_price": max_price,
            }})

        if needs_weather:
            weather_part = re.search(r"(?:thời tiết|mặc gì|nhiệt độ|mưa)(.*)", text)
            city = self.parse_city_code(weather_part.group(1)) or destination or self.parse_city_code(text)
            if not city:
                return finish("Bạn muốn xem thời tiết tại HAN, SGN hay DAD?",
                              "Chưa xác định được thành phố cần tra cứu.")
            actions.append({"name": "get_weather_forecast", "args": {"city_code": city}})

        if not actions:
            return finish("Tôi có thể hỗ trợ tra chuyến bay và thời tiết tại HAN, SGN, DAD.",
                          "Câu hỏi nằm ngoài các công cụ của lab.")
        if iteration > len(actions):
            return finish(self.summarize_observations(), "Đã đủ dữ liệu để tổng hợp câu trả lời.")

        action = actions[iteration - 1]
        observation = self.execute_action(json.dumps(action, ensure_ascii=False))
        step = {
            "iteration": iteration,
            "thought": f"Cần tra cứu bằng {action['name']} với tham số {action['args']}.",
            "action": action,
            "observation": observation,
        }
        self.trace.append(step)
        # Câu hỏi một tool kết thúc ngay; lỗi tool kết thúc, không gọi lại vô hạn.
        if len(actions) == 1 or (isinstance(observation, dict) and "error" in observation):
            answer = self.summarize_observations()
            step["final_answer"] = answer
            return answer, True
        return "", False

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            answer, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "answer": answer, "trace": self.trace,
                    "iterations": iteration, "status": "completed",
                }
        return {
            "answer": "Không thể hoàn thành trong số bước tối đa (Max Iterations Safeguard).",
            "trace": self.trace, "iterations": iteration,
            "status": "max_iterations_reached",
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== LOAD ENV ===")
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    print("Đã đọc API key:", bool(os.getenv("GEMINI_API_KEY")))

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
