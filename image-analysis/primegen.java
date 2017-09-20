import java.util.*;
import java.math.*;

public class primegen
{
  public static void main(String[] args)
  {
    BigInteger q = BigInteger.probablePrime(60, new Random());    
    System.out.println(q);
  }
}